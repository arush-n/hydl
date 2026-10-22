# The skill layer's perception: an open content schema (design, 2026-10-19)

**Status: accepted by the owner on 2026-10-19, with the decisions of section 8. All five build steps are done: the
registry (`perception/registry.py`) holds the items, blocks, gather types, creatures, attitudes and description
vectors; the v2 token types stand beside v1 (`contract/surface.py`, section 4 "As built") and the trunk reads them;
the v2 adapter (`perception/adapter.py`, section 5 "As built") fills them from the legal observation; the asks of the
simulator layer are written (`PERCEPTION_ASKS.md`); the per-tick cost is measured and set the description width to 64
(decision 1). The learner observation publishes few ids today (no inventory, no block candidates, no creature roles),
so v2 content stays unknown until the asks land.** Owner rulings that frame it (2026-10-19): the
perception schema comes before the working spine; the layers are discrete components finished independently and
combined later, so this is the skill layer's own input contract; and the architecture must stay open to mods and new
game features ("large enough for anything").

## 1. The problem

The tokens say where things are, not what they are (`contract/surface.py`, `model/observation_adapter.py`):

| Token | Today | Missing |
|---|---|---|
| block candidate | position, face | what the block is: ore or stone, wood, a door, a bench, how hard, which tool |
| inventory slot | quantity, durability, empty | what the item is: a pickaxe, food, a sword, a block |
| entity | role (always "enemy"), position, velocity, visibility, health | what kind of creature, its wind-up, where it faces, what it holds |
| geometry | solid, climbable, five named fluids, hazard | fluids beyond the list; materials |

The observation already publishes more than the adapter reads: every entity row has `entity_kind`, the entity's
facing (yaw sin and cos), `attack_progress`, `attack_phase` and `alive`
(`hytalegym/jax/combat/observation/v1/schema/contract.py:118-141`), and inventory tokens carry item ids
(`observation/v3/tokens/inventory.py:37-38`). The adapter fills one entity and leaves blocks, inventory and geometry
empty.

Raw ids cannot be the answer. An id is an integer, not a meaning, and a mod's item arrives as an id the network has
never seen. Closed vocabularies break as well. The 0.5.7 block vocabulary
(`hytalegym/worldgen/block_affordances.py`) rejects unknown gather types, and the 0.7.2 assets already use two it lacks
(`Metals`, `GoblinMetal`; 18 gather types in 0.7.2's blocks).

## 2. Principles

1. **Legal.** A feature exists only if a player could perceive or know it: by sight, in their own inventory, or by
   sound. Each feature names its legal basis. The three-world leakage test covers every new feature.
2. **Properties and relations, not ids.** Content is described by what it does, from the game's own definitions.
   Where a decision compares two things, the token carries the comparison, not the two raw values (the head-frame
   lesson of run C', README 2.16). For example, a block carries "the best power the agent's tools have against it",
   not a gather-type id. A new mod ore at a new tier then needs no new vocabulary: its tools declare their power
   against it.
3. **Fixed width.** Adding content never changes a token's width. Adding a property is a new schema version, which
   is a trunk release, and the modules are retrained with it.
4. **One source for vanilla and mods.** Hytale content is asset JSON in the same codecs for the base game and for
   mods. One registry builder reads both.
5. **Deterministic and versioned.** The registry is built offline per server version (0.7.2 by default; 0.5.7 for
   history), with its SHA-256 recorded. The adapter looks properties up by id at run time, a table gather inside
   `jit`.
6. **Generous now.** Reserve room in every content token: a short description vector, and spare property slots. The
   schema then survives the first many additions without a release.

## 3. The content registry (offline)

The 0.7.2 archive (`.../install/release/package/game/latest/Assets.zip`, 68,759 entries) holds what the schema needs:

| Content | Where | Fields that matter |
|---|---|---|
| items | `Server/Item/Items/**.json` (3,960) | `Parent` (inheritance), `Categories`, `Tags`, `Quality`, `ItemLevel`, `MaxStack`, `MaxDurability`, and the kind sections `Tool` (33; per gather type a `Power` and a `Quality`), `Weapon` (165), `Armor` (122), `Consumable` (60), `Utility`, `BlockType` (3,178) |
| blocks | the `BlockType` of block items | `Gathering.Breaking` (gather type and quality; 2,460), `Material`, `Flags`, `Opacity`, `State`, `Support`, `Interactions` (296: doors, benches, containers), `Light` (238) |
| creatures | `Server/NPC/Roles` (1,148), `Attitude` (30), `Groups` (114), `Balancing` (34) | health, damage, speed, size, attitude toward groups; roles are templated (`Type`, `Reference`, `Modify`, `Parameters`) and must be resolved |

The builder resolves inheritance and references, maps each id to a fixed-width property vector, and writes a versioned
table with its SHA-256. It reuses the repo's asset readers where they apply: the block affordances and gather types
(`worldgen/block_affordances.py`), collision and line of sight (`worldgen/surrogate/semantics.py`), the weapon catalogue
(`combat/assets/catalog.py`), and stack and durability (`jax/combat/arsenal/world_actions/crafting.py`,
`_item_properties`). Gather types become an open table that grows when the assets grow, never an enum that rejects.

Creatures, as built (2026-10-19): the 894 roles a world can spawn (`Generic` and `Variant`; 523 gameplay, 371 test)
resolve by the rules of the repo's extractor (`hytalegym/tools/content_extract/npc_roles.py`) and agree with its gen-7
table on every role. The rest was read from the 0.7.2 server's classes. An NPC's attitude toward a target comes from
the attitude table of its `AttitudeGroup`, looked up by the target's role, by `$player` for a player, or by `$self` for
a target of its own role (so two Trork brawlers fall back to their default, Ignore); then from its defaults, hostile to
players and neutral to NPCs when it names none. A table sets its attitudes in the engine's order (Ignore, Hostile,
Neutral, Friendly, Revered), so a role in two of its groups takes the later one. Groups add included groups, remove
excluded groups, add matching roles and remove excluded roles, in that order, with a case-sensitive glob. Trade is the
engine's shop actions anywhere in a role's instructions or its components; taming is the `IsTameable` parameter.

## 4. Token schema v2 (proposed widths)

Every content token keeps its position, validity and `marked` features, and gains these groups:

**entity**, the N nearest perceptible (a configuration: 16 by default, tested at 32 and above, section 8):
- relation to me: enemy, ally, neutral, animal, merchant, player, unknown. Hostility comes from the attitude tables and
  the group, not one `hostile` bit (built: the registry resolves each creature's attitude toward a player and toward
  every role; "animal" has no asset field, and the description vector carries it);
- kind properties from the registry, by the entity's role id: size, maximum health over mine, speed class, melee or
  ranged, trades, tameable. The observation's `entity_kind` cannot serve: it is a coarse kind, NPC for every entity
  (`ENTITY_KIND_NPC`, `observation/v1/schema/contract.py:246`, filled at `v1/runtime/encoder.py:140`), so the role id
  is an ask of the simulator layer (section 5);
- state from the observation: position, velocity, distance, facing (yaw sin and cos), visible, last-seen age, health
  fraction, alive, attack phase (idle, wind-up, strike, recover) and progress. A wind-up is a visible cue, the legal
  basis combat timing needs (TRAINING_PLAN 2e);
- what it holds: the kind and weapon class of a held item, if the observation can publish it legally;
- a description vector, and the reserve.

**block candidate**:
- the block's properties: breakable, ore, harvestable, soft, door, container, bench, stateful, climbable, gives light,
  supports placement, material class, hardness if the assets give it;
- relations to my kit: whether my best tool can break it, my best tool's power against it, whether the held tool can
  break it and its power. These come from my inventory's `Tool.Specs` against the block's gather type and quality;
- a description vector, and the reserve.

**inventory slot**:
- container and slot, quantity, durability, empty, active (held);
- the item's properties: kind (tool, weapon, armor, consumable, block, material, ammunition, rune, other), quality,
  item level, maximum stack; for tools, the powers over a fixed, documented set of gather families (rocks, ores, woods,
  soils, soft) plus the best power overall; for weapons, the class and the kit's reach and damage; for food, its value;
  for blocks, whether it is placeable;
- a description vector, and the reserve.

**geometry**: solid, climbable, top walkable, hazard damage, and the fluid as properties (swimmable, breathable,
damage, slows movement) instead of five names; the material class.

**self**: as today, plus the held item's kind and powers, which the agent knows.

**kit, light, route**: unchanged.

**As built** (step 2, 2026-10-19, with the description width of step 5; `contract/surface.py`, schema
`skill_tokens_v2`): entity 121 features, block candidate 103, inventory slot 106, self 36; geometry (14) and kit (15)
keep v1's. Each content token holds its v1 state,
the registry's property row (pinned in the contract: 14 creature, 15 block and 21 item properties, checked against
the registry by a test), a 64-number description vector, 8 reserve features, `known` (its id is in the registry) and
`marked` last; ids stay fields. The entity adds its facing, its attack phase (the observation's five) and progress,
its attitude toward the agent (no bit set when unknown, as for another player), `player`, and its maximum health
over the agent's; the self token adds the agent's maximum health and the held item's property row. The trunk reads
them with `EncoderConfig(token_types=token_widths(TOKEN_SCHEMA_V2))` (`model/encoder.py`). Three refinements of the
proposal above:

1. An inventory tool's powers per gather family are dropped. A family list is a closed vocabulary, which a mod's new
   gather type would break. A tool's worth shows where a decision needs it, in the block candidate's relations (the
   best carried and the held tool's power and quality margin against that block), and its description vector tells
   a pickaxe from an axe.
2. Geometry keeps v1's five fluid names until fluids enter the registry.
3. What an entity holds is a kind (unknown, nothing, block, tool, weapon, armor, consumable, other), unknown until the
   observation publishes held items.

v1 is unchanged: its surface JSON for both layouts, its widths and the trunk's default configuration are equal to
those before step 2. The v2 schema's JSON export comes with the asks of the simulator layer (step 4).

## 5. What the adapter does

It reads only the legal observation and the registry table, a constant. It fills many entity slots, block candidates,
inventory and geometry; gathers properties by id; and computes the relations (tool against block, threat against my
health) inside the same jitted function. It never reads the simulator's state.

**Asks of the simulator layer** (a contract request; that code is not this lane's; written out with file references
in `PERCEPTION_ASKS.md`, build step 4): block candidates published with their block id; each entity's role id; an
entity's held item id when a player could see it; per-cell fluid ids in geometry; the agent's own inventory, which
the learner observation does not carry today; more than 16 entity rows. The adapter maps each id to properties
itself. An attitude changed at run time (the engine's override, and
friendship between NPCs spawned together) is state, not content; whether the observation publishes it is settled with
the adapter.

**As built** (step 3, 2026-10-19; `perception/adapter.py`): `legal_tokens_v2` takes the learner observation, the
capacities and the body yaw, as the v1 adapter does. It fills the self token (v1's features; the maximum health and
the held item stay 0, unpublished) and every published entity row (up to 16 today; a larger capacity leaves the rest
invalid): position, velocity, visible, health, alive, its facing relative to the body, its attack phase and
progress, with `known` 0 and what it holds `unknown`. Block candidates, inventory, geometry and kit stay invalid.
`registry_tables` puts the registry on the device keyed by the gym's semantic ids (row 0 is the unknown row),
`lookup` maps published ids to rows inside `jit`, and `tool_relations` gives the block candidate's relations; on the
0.7.2 registry an iron pickaxe has power 0.25 against `Ore_Iron_Basalt` at margin 0, and 0.5 against `Rock_Stone` at
margin 0.6. The tables refuse a registry whose rows or description width differ from the schema's, or whose ids
collide.

## 6. Versioning

The schema id is `skill_tokens_v2`. v1 stays beside it: the A2 reference runs (A2 to C') were trained on v1 widths and
must keep reproducing. The trunk is trained on one schema, and every module records the trunk's SHA-256
(already built, `model/skill_module.py`). A width change is a new schema, a new trunk and retrained modules, released
together with regression tests. Training is paused, so no module exists yet to retrain: v2 is cheapest now.

## 7. Tests (when built)

- legality: the three-world leakage test over every new feature;
- the registry is deterministic: the same archive gives the same SHA-256;
- an open vocabulary: a synthetic mod item and block (new ids, new gather type, defined in asset JSON) give valid
  tokens with no code change, and their relations are right;
- widths never change when content is added;
- relations: an iron pickaxe against iron ore gives the power the assets declare.

## 8. Decisions (owner, 2026-10-19)

1. **Strength first, within 30 Hz.** "we need the lower level to learn as best as it can, we just need it to operate
   at 30hz so there is time if we need, we need the model to be as strong as possible while maintaining a good speed."
   So every content token carries a **description vector** now: a fixed-length embedding of the asset's name,
   description, tags and categories, computed offline by a pinned text encoder and reduced to a fixed width by a
   projection fitted on the 0.7.2 corpus. At run time it is a table gather, so it costs width, not encoding time. The
   width (proposal: 32) is set by the latency measurement of step 5. **Set to 64** (step 5, 2026-10-19,
   `bench/bench_perception.py`): the width changes the trunk's cost by less than the noise (batch 8, 64 entities:
   p95 1.62, 1.61 and 1.65 ms at 16, 32 and 64 numbers; v1 1.58 ms), while 64 numbers keep 80% of the item
   embeddings' variance (66% at 32) and agree with the full embedding on 73% of nearest neighbours (60% at 32);
   creatures 84% and 78% (71% and 68%). The trunk's model width (64) bounds what a wider vector could carry.
2. **Entities scale with the world.** "add support for 32 or higher but work on 16 right now": the entity capacity is a
   configuration, 16 by default, with 32 and above covered by the tests and the latency measurement.
3. **0.7.2 only.** 0.5.7 stays history; the registry builder is keyed by version, so another can be added.
4. **What is legal about others** (health bars, held items, wind-ups) follows the game's own rules, checked against
   the live 0.7.2 server before those features are filled.

## 9. Build plan (no training; each step tested before the next)

1. **The registry** (`perception/registry.py`, offline): read the 0.7.2 archive; resolve item `Parent` chains and NPC
   role references; build the item, block and creature property tables, the open gather-type table and the
   description vectors; write the tables with their SHA-256. Tests: determinism, inheritance, an open vocabulary (a
   synthetic mod's item, block and gather type).
2. **The v2 token types** beside v1 (`contract/surface.py`), and an encoder configuration that reads them.
3. **The v2 adapter** (`perception/adapter.py`): the legal observation and the registry in, v2 tokens out; many
   entities; the relations (my tools against a block, a creature's threat against my health). Tests: legality, fixed
   widths, the relations, the open vocabulary.
4. **The asks of the simulator layer** (section 5), written as a contract request for its owners.
5. **The cost per tick** at 16, 32 and 64 entities against 30 Hz, which sets the description width.

## 10. Token schema v3: space, the agent's own state, and everything legal (owner, 2026-10-20 night)

The owner set the order: expand the encoder to every piece of legal information we plausibly can before any further
architecture work, and finalise how the network reads space, its own state and the world. The inventory behind it
(four read-only surveys of the encoder, the simulator's legal observation, the live wire and the action space) found
the trained trunk reading one opponent token and one self token, v2 unused at run time, and about thirty published
fields read by neither adapter. The owner's four decisions:

1. **Space: the body and the head frame on every spatial token.** One function (`state/frames.py`) gives every
   spatial token the same `where` block: its position in the agent's body frame (forward, right, up, in blocks over
   16), the log of its distance (`log1p(d) / log1p(64)`), its bearing in the body frame and in the head frame (sine and
   cosine), and its elevation from the eye relative to the head's pitch (sine and cosine). Locomotion reads the body
   frame, gaze and aim the head frame; run C' showed the head frame is what made looks generalise (README 2.16). World
   rows (projectiles, hazards) are rotated into the body frame first. The self token carries the head's offset from
   the body, its pitch and the body's compass heading, which the agent knows.
2. **Memory: last-seen tokens.** An entity that leaves view stays an entity token, `remembered`, with its age, moved
   into the current body frame each tick by the agent's own motion (its position, when published). Bounded by the
   entity capacity; legal, as a player remembers.
3. **Its own actions: held-for counters.** One `act` token per action head: the head, the choice of the last tick
   (as a fraction and whether neutral) and how many ticks it has been held. The engine never shows a hold's progress,
   so this is what lets a skill time a charge, a draw or a release.
4. **Simulator asks go to the coordinator** (S1 to S9: perception_v2 on by default, role ids, entity rows 1 to 15,
   held items and armour, block and fluid ids, the missing legality gates on projectiles, areas and the target's
   ability rows, other entities' movement and statuses, occluders and audio). Each v3 token has its seam ready; a
   seam reads zero, with `known` 0, until its field is published.

**Legality.** Unchanged principle 1. Where the simulator publishes a row without a perception gate (projectiles and
areas: distance only; the target's ability and weapon rows), the adapter applies the view sector it can compute, and
leaves out what it cannot gate (the target's ability rows) until the simulator gates them (S7).

**The token types** (`contract/tokens_v3.py`; capacities are a configuration, these the defaults):

| Type | Count | What it carries | Source today |
|---|---|---|---|
| self | 1 | velocity in the body frame, speed, grounded, fall speed, fluid and submersion, the 23 movement states, control lock, damage recovery, alive, head offset and pitch, compass heading, health and maximum health, 7 resources with availability, 8 defense values, attack executing and cooldown, the held item's properties and description | `self_f32`, `resource_f32`, `defense_f32`, `actor_world_f32`, `movement_state_f32`, `weapon_i32`; registry |
| act | 21 | per action head: which head, last choice (fraction, neutral), ticks held | the actor's own output (carry) |
| kit | 16 | per own ability slot: which slot, duration, cooldown, active progress, affordable, legal, 7 costs; hold count and thresholds, charged, projectile speed and gravity, from the compiled loadout by ability id | `ability_f32/i32/legal` (own row); loadout |
| status | 8 | own effects: remaining, cycle, damage, healing, resource delta, speed multiplier | `status_f32/i32` (own row) |
| entity | 16 | where; velocity; facing relative to my body and toward me; visible, remembered and age; health and alive; attack phase, progress and index; kind, attitude, player, held item (seams); description | `entity_f32/i32`; perception_v2; registry; memory |
| projectile | 16 | where; velocity; closest approach (time, distance); hostile, mine; damage, radius, age, lifetime | `projectile_*` (view-sector gated here) |
| hazard | 8 | where; half extents; intensity, duration, damage rate, activation, hostile | `hazard_*` (view-sector gated here) |
| geometry | 44 | where; kind; clearance; the cell flags (solid, opaque, fluid, damaging, climbable, bouncy, trigger, protrudes, movement); dynamic blocked; traversal edges (each its destination's displacement in the body frame, cost, kind, flags; the gym's destination index cannot be resolved by the trunk, section 11); detail boxes; light (sky, block RGB, tint) | `world_geometry`, light tokens |
| block_candidate | 16 | where; affordances; gather type bits; required quality; registry properties, tool relations, description (block id seam, S5) | the action surface's block candidates |
| inventory_slot | 76 | container, slot, quantity, durability, active; item properties and description | perception_v2 inventory, or the env's inventory tokens |
| recipe | 16 | the recipe candidate's encoding | the action surface's recipe candidates |

Every type ends in `marked`. v1 and v2 stay as they are: the A2 reference trunk reads v1.

### 10.1 Memory: three timescales, three owners (owner, 2026-10-20 night)

"It does not need motion over time if there is a memory tensor encoding past states": no hand-made trails or
trajectories; motion is inferred from memory.

| Timescale | Where | What it keeps | Cost |
|---|---|---|---|
| ticks to seconds | the trunk's latent memory (`model/encoder.py`, `memory_slots`): S slots that pass through the trunk with each tick's tokens and are blended into the next tick's by a learned gate | what the network learns to keep: motion, recent events | S tokens of attention; fixed |
| seconds to minutes | the spatial store (`state/memory.py`): world-anchored last-seen entities, geometry cells and block candidates, read back each tick as `remembered` tokens in the current body frame | exact positions of what left view; evicted by priority (hostiles and points of interest kept longer), not age alone | fixed capacities |
| beyond minutes, across sessions | the decision layer and the LLM (not this lane): the neural layer reports salient events onto the blackboard; the upper layers index them (place, entity, time) and hand back what matters, as request arguments or as a small set of retrieved landmark tokens | everything worth keeping long | query-driven, off the 30 Hz path |

The first two are arrays in the neural layer's carry, saved and loaded with it.

### 10.2 Certification of the encoder (verified as built; nothing is "done" without its evidence)

| Gate | What must hold | Status |
|---|---|---|
| E1 frames | `where` gives the body and head bearings and the elevation of known placements; `world_to_body` is the gym's frame | **confirmed**: test_sa_tokens_v3.py (`test_where_reads_the_body_and_the_head_frame`, `test_world_to_body_uses_the_gyms_frame`) |
| E2 schema | every type ends in `marked`, has unique features, spatial types lead with `where`, remembered types carry `remembered` and `age`; widths pinned | **confirmed**: test_sa_tokens_v3.py |
| E3 trunk | the trunk reads v3 and ignores padding and capacity | **confirmed**: test_sa_tokens_v3.py |
| E4 no regression | the trunk without memory is unchanged; the trained C' trunk still loads | **confirmed**: encoder, freezing, skill-module, generalist and spine tests, 70 passed |
| E5 decode, per type | each type's rows match a known placement on a real observation: self, kit, status, entity, projectile (archery range); geometry, block candidates, recipes, inventory (an environment that publishes them) | **partly confirmed** (test_sa_state_senses.py): entity placement in the body and head frame, the kit against the engine's loadouts (the bow's tier speeds, the battleaxe's 0.85 s charged hold, the sword's reach), the agent's own arrow (ahead, mine, flying away), geometry through the gym's own world-token encoder (position, flags; a traversal edge as its destination's displacement in the body frame, 2026-10-21), the action history, packing. 2026-10-21: hazards (place, half extents, damage rate, the view gate), a block candidate through the gym's own `encode_block_action_candidate_policy_view` (place, affordance bits, gather type, tool quality), the status flag bits (pinned to the gym's constants), the engine's `hostile` flag on entity rows, the affordance names and the recipe width pinned to the gym's. A coverage audit (every feature a builder writes, one tick with every optional input) leaves unwritten only the reserves, the self row's `max_health_log` (unpublished, S17), and the block candidate's registry columns (the block id seam, S5); inventory reuses v2's tested rows (test_sa_perception_v2.py). Recipes are read as the environment's embedding, whose encoder is fixed at a seed and never trained (`arsenal/environment.py`, `_default_recipe_candidate_encoder`): when an arena publishes recipes the trunk should own those parameters (open) |
| E6 legality | a hidden entity (behind, or at another distance out of view) leaves every token identical; projectiles and hazards out of the view sector are dropped | **confirmed** for entities and projectiles (test_sa_state_senses.py: a target behind at 5 or 10 blocks gives identical tokens of every type; a projectile behind is dropped, one ahead kept while the head looks there); hazards use the same gate, untested |
| E7 memory | the latent memory carries information across ticks and resets; the spatial store keeps, moves and evicts rows as specified | **confirmed**. Latent (test_sa_tokens_v3.py): two pasts read differently; the initial slots are a clean reset; padding-invariant; no parameter change without memory. Stores (test_sa_state_memory.py, with a synthetic perception channel): an entity seen 6 blocks ahead, hidden after the agent walks 2 blocks and turns 90 degrees, reads 4 blocks to the right, remembered, not visible, one tick old, and is not perceived twice; a tagged place never seen is a placeholder block row at the right place with its priority; a tagged entity survives an overflowing store while the oldest untagged ones are evicted; without the agent's position the stores do nothing; reset clears; a carry saves and loads exactly. **Reopened and re-confirmed 2026-10-21**: the first confirmation used motionless entities and checked positions only, and the store kept every column but `where` in the body frame of the tick it was seen, so after a turn a remembered entity's velocity and facing, and a remembered cell's edges, pointed the old way. Now stored in world axes and turned on reading: an entity seen moving along +x and facing -z, remembered after a 90-degree turn, reads that motion to the right and that facing straight ahead; a remembered cell's edge turns with it (22 v3 tests). Open: the stores on a real gen-7 perception channel (S1); an entity's own velocity across ticks (S16: the gym scales entity and self velocity by different speeds) |
| E8 cost | the full v3 tick (adapter and trunk, memory on) within the 30 Hz gate (p95 at most 15 ms at batch 8) | **confirmed at the default size** (scratch bench on real archery-range observations, jitted adapter and trunk, load 12 to 14): width 64, 2 layers, 8 memory slots (213,184 parameters, 346 tokens an agent): p95 2.59 ms at batch 1, 5.20 ms at batch 8. A larger trunk (width 128, 4 layers, 16 slots, 995,456 parameters): 4.06 ms at batch 1, 25.56 ms at batch 8, over the gate (inside the 33 ms tick); a larger trunk needs the speed-ups first (caching the slow tokens, a slower trunk rate). Re-measured 2026-10-21 after the geometry (185) and status (22) widths changed, load 8 to 11: p95 1.38 ms at batch 1 and 4.86 ms at batch 8 at the default size; the larger trunk 2.82 and 9.63 ms (its earlier 25.56 ms was under a heavier load). Re-measured 2026-10-21 after step 1 (the registry, the 11-wide persona token), the state layer's tick plus the trunk with conditioning and 8 memory slots on duel observations, load 10: p95 5.02 ms at batch 8 |
| E9 live consistency | the native assembly's tokens equal the simulator's for the same state | **partly** (2026-10-21, section 10.5): the state layer's real tick ran on the live 0.7.2 server (bridge 6DFD8682) and filled every type the live path publishes (self, kit, status, entity, geometry with light, act, persona, affect), with the same scales as the simulator on the shared columns. Token equality for the same state is still open: it needs a live trajectory replayed in the simulator from the same start. The 204 saved live observations replay offline with the simulator's exact structure (63 leaves, the same shapes) through the state layer, the trunk and the generalist head, all outputs finite (E11) |
| E10 bounded inputs | every feature within ±5 on rollouts (running normalisation clipped at ±5 in OpenAI Five) | **confirmed on active rollouts, reopened and re-measured 2026-10-21** (section 10.5). The first measurement's rollouts were nearly static: its scripted sword body only swings at a target already in reach and never moves or turns, so the agent's speed was always 0 and one lane in eight saw the target, at one place. Re-measured on six runs where the agent acts at random (legal, held) against an attacker: the gen-5 and gen-7 duels, the gen-7 channel run, a gen-7 Region world (55,698 geometry rows) and two live runs. The largest value is 1.94 (a projectile's place beyond the 24-block scale); every other column is within 1.48. Still open for hazards, block candidates and recipes (published by no run). A block candidate's `required_quality` is the raw tier, not a fraction |
| E11 the state layer and the trunk together | on real observations, each tick's tokens read by the trunk with its latent memory and the persona and affect conditioning, and the head on the trunk | **confirmed** (2026-10-21, `test_sa_tandem_v3.py`, 7 tests on a duel rollout where the agent acts at random against a chaser): the tokens' widths are the trunk's; the opponent reaches the trunk; padding and a larger capacity leave the output unchanged; the latent memory carries the past and a reset lane equals a fresh start; another persona changes the output and a trait the body has no mechanism for (a value) changes nothing; a state and a trunk saved mid-episode continue exactly; a trunk migrated from an older layout reads real tokens as before; the generalist head runs on the v3 trunk. The 204 live observations, replayed: the same, all finite. **Integrated** 2026-10-22 (step 1b, `VIRTUAL_HUMAN.md` section 13): the generalist reads every type; every entity row carries its id; `runtime/state_tick.py` ticks the state layer and the trunk for the bodies, and a request points at a perceived or remembered entity (`test_sa_state_tick.py`, 5 tests). The trainers' token kernel stays on v1 until training resumes |
| E12 the picture against the world | on real rollouts, the state layer's picture of the world against the simulator's true state | **measured** (2026-10-21, owner: "does it properly model the world"; `test_sa_belief_truth.py` on the duel, the opponent chasing the evading agent; a gen-7 Region run for terrain): the seen opponent is within 0.07 blocks of its true place (median 0.04; the channel rounds to 0.1); the heading is the body yaw; the speed column is a fixed 1/7 of the true speed; 108,583 remembered terrain cells map back to where they were seen within 0.000003 blocks. A remembered moving entity stays where it was last seen: its error equals its movement since (difference p95 0.05 blocks), median 0.75 blocks under 10 ticks out of view, 3.6 at 10 to 30, 10.9 at 30 to 90, 35.8 at 90 to 300. So the present and the static past are exact, and nothing predicts where a hidden thing has gone: N2, to come from the latent memory's training (N1), the owner having ruled out hand-made motion. The trunk is untrained, so it models nothing beyond its inputs yet |

### 10.3 Two stores: the spatial map and the salient store (owner, 2026-10-20 night)

"We need to store important information and spatial information separately ... entities, certain key blocks and some
other features could be important and need to be remembered/targeted. These important tags should be able to be
passed down by the middle layer." The neural layer's memory is two stores with different rules:

| Store | Holds | Kept by | Read back as |
|---|---|---|---|
| spatial map | the shape of space the agent has seen: geometry cells (solid, walkable, climbable, fluid, hazard), anchored to the world | recency and nearness; lossy is acceptable, and it is bounded | `geometry` tokens marked `remembered`, with their age |
| salient store | the things that matter, with identity: entities, key blocks (ores, doors, benches, containers, beds), dropped items, places; and anything the middle layer tags | importance: the layer's own rules (a hostile entity, a rare or useful block) and the middle layer's tags; a tagged item is pinned until its tag expires or is withdrawn | `entity` and `block_candidate` tokens marked `remembered` (and later a `landmark` type for places and items), each with a stable handle, its tag and its age |

**Tags from the middle layer.** A tag is a request, through the protocol that already exists: a scope-less verb
(`remember`, owning no scope, so it never blocks and never drives a head; the combiner already accepts such verbs) whose
argument is the thing (an entity pointer, a position or a cell) and whose scalars give its label and priority. The
neural layer pins the tagged thing in the salient store, marks its tokens with the tag, and reports it back with its
handle, so a later request can point at it ("go to the chest you were told about", "attack the tagged Trork"). The
verb and its label vocabulary are the catalogue's to add (contract/catalogue.py, this lane); the middle layer decides
when to send them.

**Why separate.** Space is bulk and changes little; it can be summarised and evicted by distance. Salient things are
few and decisive; losing one (where the ore was, which Trork hit me) costs more than a thousand cells, so they are kept
by importance, never evicted by bulk.

### 10.4 Progress log (newest last)

- 2026-10-20 night. Schema v3 (`contract/tokens_v3.py`), the frame convention (`state/frames.py`), the trunk's v3
  support and its latent memory (`model/encoder.py`, `memory_slots`, off by default), the action history
  (`state/action_history.py`) and the v3 adapter (`state/senses.py`) written. E1 to E4 and E7 (latent) confirmed
  by tests (section 10.2). The adapter ran once on the archery range with plausible values (E5 open).
- Found while certifying: `combat_cover` (the coordinator's arena, `tasks/arenas.py`) publishes no world geometry
  tokens, no light and no block or recipe candidates (all unavailable at reset), though it binds a geometry provider;
  only the inventory tokens are present. So no environment in reach gives the learner its local geometry today;
  geometry and block decoding (E5) are certified on the gym's own token builders until one does. Sent to the
  coordinator.
- Certified the adapter: E5 (partly) and E6 confirmed by test_sa_state_senses.py, 8 passed (section 10.2).
- E8 measured: the default v3 tick is 5.2 ms p95 at batch 8 (gate 15 ms).
- The stored memory built (`state/memory.py`: the spatial map and the salient store, tags, placeholders) and the stateful tick (`state/layer.py`: history, stores, packing, save and load); the remembered types gained `tagged` (entity 154, geometry 161, block candidate 123 features). E7 confirmed: 19 v3 tests pass (test_sa_tokens_v3.py 6, test_sa_state_senses.py 8, test_sa_state_memory.py 5).
- 2026-10-21. The encoder audit and the literature check (section 11). Two defects of this lane found and fixed: a
  traversal edge named its destination by the gym's token index, which the trunk cannot resolve (now its displacement
  in the body frame; geometry 161 -> 185 features); the stores kept every column but `where` in the body frame of
  the tick it was seen (now an entity's velocity and facing, and a cell's edges, are stored in world axes and turned
  on reading). E5 and E7 re-confirmed: 22 v3 tests (tokens 6, adapter 9, memory 7). Simulator gaps S11 to S16 sent
  to the coordinator (`PERCEPTION_ASKS.md`).

- 2026-10-21. Encoder completion: the status token reads the flag bits (status 16 -> 22 features); the entity token
  reads the engine's `hostile` flag in place of `health_ratio` (an entity's maximum health is not perceivable), and the
  salient store ranks by it while the role is unknown. E5 widened (hazards, block candidates, status flags, pins) and
  E10 measured (section 10.2); E8 re-measured. 27 v3 tests. Excluded with reasons: v1's attack sequence index (stays
  0 on gen 7 over 120 ticks of sword attacks: not live), the ability `requirement_bits` (a static mask; `legal`
  carries whether it applies) and `evidence_level` (data provenance, not a percept). Deferred: closing speed (derived
  from place and velocity, no new information) and time to reach (needs S16).
- 2026-10-21, owner: **no explicit voxel map** ("idk if we need the explicit voxel map we will see later; personally
  i think a compressed latent space would be sufficient and i am fine with it if it is fast enough"). Space is held
  by the stores' remembered tokens and the trunk's latent memory slots; E8 shows them fast (4.86 ms p95 at batch 8).
  The voxel channel (section 11.3, N6) is dropped until training shows a need.
- 2026-10-21, owner: the architecture paused and settled in discussion (`VIRTUAL_HUMAN.md`): a state layer holds
  everything about the agent, in memory, read by every layer; this document's tokens are the body's view of its
  world and body part. The persona and affect code is a proposal to reshape under it; nothing of it is committed.

### 10.5 The encoding on the simulator and the live server (2026-10-21)

The owner: "start to test the encoding in the sim and the real server to see if we get the information we need".

**How** (`../bench/encoding_audit.py` and `../bench/live_encoding.py`; the runbook is `../bench/README.md`). The
state layer's real tick (`layer.step`, with the 0.7.2 content registry's tables) ran on rollouts in
which the agent takes random legal actions, each held for about 8 ticks, against an opponent that attacks. For every
column of every token type it counted the valid rows, how often the column is non-zero, its range, and its distinct
values. The runs:
- **duel:** the open-flat duel (`tasks/duel_arena.py`) on a private gen-7 render, sword, shortbow and battleaxe,
  8 lanes x 300 ticks; a chaser (close to 1.5-2.5 blocks, swing in reach) plays the opponent. The same on gen 5.
- **channel:** the single-actor environment on the gen-7 render with `perception_v2` on (position, body yaw,
  inventory, entity kind and held item), open field and the gym's wall geometry.
- **region:** a Region world (`worlds/region.py` `region_scene`, the pinned library `region-library-v3-288`) on the
  gen-7 render with the channel on: real terrain, its geometry and light tokens, the built-in Trork target active.
- **live:** the live 0.7.2 server (pinned bridge 6DFD8682, `HytaleEnv` native backend, `kill_trork` on the flat
  world, the Kweebec_Razorleaf body, the target active), the tick run in-process on each live observation; 74 and
  204 ticks (both runs ended on a native decode error, below; after the coordinator's fix, 745a0cf, the same program ran 300 ticks, all valid, with the same columns carrying information and the dodge's invulnerability too).

**Columns carrying information** (varying + constant non-zero, of all; valid rows):

| Type | duel (gen 7) | channel (gen 7) | region (gen 7) | live 0.7.2 |
|---|---|---|---|---|
| self | 99+19 of 173 (7,200) | 90+18 (4,800) | 35+81 (1,200) | 30+83 (204) |
| act | 25 of 26 | 25 | 25 | 25 |
| kit | 24+1 of 39 | 16+1 | 14+1 | 14+1 |
| status | 0+2 of 22 | 0+2 | 0+2 | 0+2 |
| entity | 27+4 of 158 (2,480) | 22+4 (4,451) | 31+4 (1,200) | 27+5 (204) |
| projectile | 19+1 of 23 (382) | no rows | no rows | no rows |
| geometry | no rows | no rows | 71+4 of 185 (55,534) | 13+10 (2,147) |
| inventory slot | no rows | 67+8 of 106 | 1+74 | no rows |
| hazard, block candidate, recipe | no rows | no rows | no rows | no rows |
| persona, affect | 10 of 11, 8 of 9 | 10, 7 | 10, 8 | 10 (one agent: constant), 8 |

**What arrives, everywhere it is published.** The body's motion, movement states, head and heading, health,
stamina and the other resources, defense, attack state; the held item with its registry properties and description;
the kit; another entity's place, motion, facing, health and attack phases; projectiles; terrain cells with clearance,
traversal edges and sky light; the remembered cells and entities (where the channel gives the position); the action
history; the persona and the emotional state. The live server fills the same columns as the simulator, at the same
scales (the self speed column peaks at 0.928 in both).

**What does not arrive** (the gaps, by owner):

| Gap | Where | Owner |
|---|---|---|
| FIXED (coordinator, 745a0cf): the live decoder refused a frame (`ValueError: native defense evidence is incomplete`) when a hit's knockback was queued while the agent was airborne after a jump; the decoder now sums the projection on the host. The same random program then ran 300 live ticks, all valid (2026-10-22) | live | coordinator |
| FIXED (coordinator, 3f8aded): guard on gen 7. The mask marked `guard_off_on` legal and the step rejected it, so guard never engaged: a re-entry patch closed the guard choice while a guard was held, dropping it on its second tick. Now a held guard engages on tick 3, as live; `defense_guard_active` varies in the duel and the Region world (audits re-run on a render of 3f8aded) | gen-7 render | coordinator |
| `perception_v2` exists only on the state-taking encoder; the duel's multi-actor path and the live native assembly carry none. There, no position (so no memory), no inventory, no entity kind or held item. The observation has no other legal position (`actor_world` is medium, submersion and drop) | duel, live | coordinator (asks 1, 2, 5 and the live channel) |
| The entity's role id is 0 in the channel (its seam), so every `kind_*`, `attitude_*` and `known` column reads 0 | all | coordinator (ask 1) |
| One entity row only, in every run | all | coordinator (ask 3) |
| The Region fixture gates no entity row by the view sector or the sensor range (`project_world_runtime` omits both), so there the agent perceives what is behind it: only line of sight and the 24-block radius hide an entity (`HytaleRL/examples/jax_arsenal_world_benchmark.py`, `geometry_arsenal_world_capabilities` called without them) | region | coordinator: filed; that file has another session's uncommitted edits, so the fix waits for them or the owner |
| No block candidates: the 0.7.2 Region library has no block semantics (only the 0.5.7-captured `region-library-pilot-v2` does); live needs the world-action adapter, which has not run live | all | coordinator (asks 4, S5) |
| No hazards: only ability-spawned areas make them (the flame staff's fire trap) | all | coordinator (S10) |
| Collision boxes (`box_*`) read 0 in the Region world and live; the live flat world also sends no traversal edges | region, live | coordinator |
| `moving_jumping` and `moving_sprinting` never set, though the agent leaves the ground (falling is set) and picks the sprint gait; sprint matches #126 (sprint capped at run live) | all | coordinator (to confirm) |
| My maximum health (S17) | all | coordinator |
| FIXED (5b4a7ca): damage taken was credited to an attacker only in its sweep and within 3.0 blocks. The duel's sword lands as the attacker enters recovery, from up to 3.48 blocks, so the feelings toward it never rose (they did toward the Trork, which hits during its sweep, in the Region world and live). Now a sweep, else a recovery, within the measured melee reach of 3.75 blocks (`THREAT_REACH_BLOCKS`); the duel's hostility, trust and anger columns fill (test A4b) | duel | this lane (`state/affect.py`) |
| The status token drops the effect's id, its source entity and its damage cause, which the observation carries (`status_i32`): the agent sees a status's effects but not what it is or who applied it. The duel's permanent status is the game's infinite `Immunity_Environmental` (speed 0.5 is the encoder's multiplier over 2, so x1). Planned with the effects registry in step 7 (`VIRTUAL_HUMAN.md` section 13) | all | this lane (the token) |

The coordinator filed every gap in its lane on 2026-10-21 and is working on two: the live decoder failure (it will
replay these action sequences live once fixed) and gen-7 guard; the channel's reach extends #92 and waits on the
owner's scope.

The duel and Region numbers above are the repo tool's, re-run on a gen-7 render of 3f8aded, after the attribution fix
and the guard fix. With guard engaging, a few columns moved: the Region target's health and the agent's signature
energy no longer change in this run (inferred: more hits are blocked). The live runs predate the attribution fix
(inferred unchanged: their attacker hits during its sweep, and the new reach moves values, not which columns carry
information); the guard fix is the simulator's only. The live captures (raw leaves and actions) are kept in this
session's scratch for the coordinator's replay of the decoder failure.

## 11. Is the encoding enough? The literature and a field-by-field audit (2026-10-21)

The owner: "go a bit deeper into the encoder and ensure that we are storing all the information that we will
plausibly need ... look at general papers around this type of thing and also robotics". Three literature surveys
(game agents, robotics, and individuality; the last is in `IDEAS.md` Idea 2), each paper's existence checked by
search, lessons from abstracts unless the paper's text was read; and an audit of every field of the learner
observation against `state/senses.py`.

### 11.1 Verdict

- **As an input specification, v3 meets published practice.** The game agents that act on structured state (Dota 2
  with Large Scale Deep RL; AlphaStar; Emergent Tool Use from Multi-Agent Autocurricula) use what v3 uses: a token per
  entity, a mask for what is not perceived, the last-seen state of what left view, the previous action. Legged and
  navigating robots keep control inputs in the base frame and maps in the world frame, read back egocentrically
  (Active Neural SLAM, Cognitive Mapping and Planning, Neural Map, Hydra): v3's body and head frames over
  world-anchored stores, with an exact pose where a robot must estimate its own. The vision-language-action models
  (RT-1, RT-2, OpenVLA, Octo, pi0) encode far less (one or two images, little proprioception, 1 to 5 Hz) and are no
  template for a 30 Hz body.
- **What is missing is of three kinds, none of them more columns of the same kind:**
  - events the simulator never shows the learner: damage taken and dealt, and sound;
  - belief over what is remembered: v3 keeps an age, but no confidence and no predicted place;
  - supervision in training: a critic that sees the whole state, and an auxiliary loss that makes the memory hold
    hidden state.

  The largest reported gains come from the last kind. Hide-and-seek's run with a masked critic never passed stage 3
  of 4. Legged robots learn hidden terrain and contact from history only when a privileged teacher or a
  reconstruction loss supervises it (Lee et al. 2020, RMA, Miki et al. 2022).
- **Memory size.** Small memories are competitive:
  - in AlphaStar Unplugged's behaviour cloning, no memory scored above an LSTM;
  - in Memory Gym, a GRU beat Transformer-XL on the endless tasks.

  Recurrent state alone does not hold terrain: Neural Volumetric Memory beat a GRU on stepping stones, 14.4% against
  2.2%. So the split is right: explicit stores beside a small latent memory.

### 11.2 Every field of the learner observation, and what v3 does with it

| Field | v3 | Why |
|---|---|---|
| `base.self_f32` (16) | self | all read |
| `base.self_i32`: tick count, attack sequence index, pending attack index, motion timing profile | not read | the tick count is an episode clock, not a percept; the attack sequence and pending attack are the v1 melee chain (gen 7's chains show as each ability slot's active progress, in `kit`), to add if a check shows them live on gen 7 (N10); the timing profile is engine configuration |
| `base.target_*`, `base.combat_*` | not read | the designated target's row again; a target reaches the neural layer as a request's pointer to its entity token |
| `base.entity_*`, `projectile_*`, `hazard_*` | entity, projectile, hazard | all read; projectiles and hazards gated by the view sector here |
| `base.terrain_*` (v1) | not read | superseded by v3's `world_geometry` |
| `base.traversal_*` (v1) | not read | a route planner's waypoints (path cost, travel time, confidence): a plan, not a percept; a route reaches the neural layer as a request argument |
| `base.interaction_*` (v1): doors within 6 blocks | not read | the geometry token's door flag and `where` carry the same; no arena publishes them (the duel arena's masks are all 0) |
| the action masks (skill, ability, jump, guard, dodge, door) | the heads | masks belong to the action heads; `kit` carries each ability's `legal` |
| `weapon_i32`, `resource_*`, `defense_f32`, `status_*`, `actor_world_*`, `movement_state_*` | self, status | all read |
| `ability_f32`, `ability_i32`, `ability_legal` | kit | `evidence_level` and `requirement_bits` not read (N10) |
| `world_geometry`, the light tokens | geometry | read; published by no arena (S10) |
| the action surface's block and recipe candidates | block_candidate, recipe | read |
| `perception_v2` | self (heading), the stores (position), entity (kind, held item, role), inventory | read; off by default (S1) |
| in the simulator, not in the learner observation | none | damage and hit events exist only as a trace channel (`v3/native/channels/events.py`; S11); sound exists as `jax/world/audio` with no consumer (S12, part of S9); no item drops (S13); no clock or weather, the light being a surrogate sky light from terrain height (S14); no team or group of an entity (S15); entity and self velocity in different units (S16) |

Two defects of this lane, found by the audit and fixed (section 10.2, E5 and E7):
1. The gym's traversal edges name their destination by an index into its own token list, which the trunk cannot
   resolve, since a token carries no index. The adapter now gives the destination's displacement in the body frame
   instead.
2. The stores kept body-frame columns (an entity's velocity and facing, a cell's edges) in the frame of the tick they
   were seen. They are now stored in world axes and turned on reading.

### 11.3 The gaps, by owner

**Neural layer (this lane).**

| # | Gap | Evidence | Plan |
|---|---|---|---|
| N1 | Supervised memory: a critic on the full state (training only; the actor stays legal) and an auxiliary head that decodes hidden entities' places from the latent slots | Asymmetric Actor Critic; hide-and-seek; Lee et al. 2020; RMA; Miki et al. 2022 (reconstruction weight 0.5); MERLIN; Memory Gym (Transformer-XL won only with a reconstruction loss) | the trunk gains a privileged token set read by the critic only; part of the training design, before reinforcement at scale |
| N2 | Belief over the remembered: a confidence and a predicted place beside the last-seen one | Cognitive Mapping and Planning (a confidence channel); SORT (a Kalman prediction per track); object permanence needs supervision on occluded objects (Tokmakov et al. 2021) | the owner ruled out hand-made motion ("a memory tensor encoding past states"), so N1's auxiliary loss comes first and a dead-reckoned place only if it fails; an entity's own velocity waits on S16 |
| N3 | Bounded inputs: every feature within ±5 across a rollout battery; symlog on heavy-tailed scalars | OpenAI Five (running normalisation, clip ±5); DreamerV3 (symlog) | gate E10: measure each feature's range on rollouts of every arena |
| N4 | Closing speed and time to reach for each entity (its radial velocity; ticks until it is within my reach) | time to contact (Lee 1976); projectiles already carry their closest approach | two entity columns |
| N5 | Noise, latency and dropout in training | Miki et al. 2022; Agarwal et al. 2022; Rudin et al. 2021 | training configuration; needed before the live port |
| N6 | Local terrain as a fixed grid around the feet, or a small 3D network over the spatial map | Lee et al. 2020; Miki et al. 2022 (208 height samples); Rudin et al. 2021; Neural Volumetric Memory | dropped (owner, 2026-10-21): the stores and the latent memory are the compressed space; revisit if training shows a need |
| N7 | An event token type, a sound token type, item rows | MineDojo (`damage_source`); Neural MMO (damage); AlphaStar (`alerts`); sound in ViZDoom | wait on S11, S12, S13 |
| N8 | Who did what to me: per-entity counters in the salient store (hits taken from it and dealt to it, the age of its last hostile act) | ToMnet; LIAM; OpenAI Five ("is this unit attacking me") | waits on S11; also what grudges and trust are made of (`IDEAS.md` Idea 2) |
| N9 | Fourier features on the `where` block | Fourier Features Let Networks Learn High Frequency Functions; Perceiver; thin evidence in RL | a sweep during training, not before |
| N10 | The kit's requirement bits; the combo position if live on gen 7 | none | two small columns after a check |

**Simulator (the coordinator; sent 2026-10-21).**
- S11: damage and hit events (high).
- S12: sound (medium-high).
- S13: item drops.
- S14: clock and weather.
- S15: an entity's team.
- S16: one velocity unit for entity and self rows.
- S17: my own maximum health, and a block's break progress (both shown to a player).

All are listed in `PERCEPTION_ASKS.md`. The coordinator filed them and is confirming the new scope with the
owner.

**Upper layers (not this lane).**
- A teammate's intent, and communication: the blackboard.
- A places layer in long-term memory (Hydra's topology).
- The persona's values (`IDEAS.md` Idea 2).

### 11.4 Sources (title, year, arXiv id or venue)

- Dota 2 with Large Scale Deep Reinforcement Learning (2019), arXiv:1912.06680.
- Grandmaster level in StarCraft II using multi-agent reinforcement learning (2019), Nature 575; AlphaStar Unplugged
  (2023), arXiv:2308.03526.
- Emergent Tool Use From Multi-Agent Autocurricula (2019), arXiv:1909.07528.
- MineDojo (2022), arXiv:2206.08853. Neural MMO (2019), arXiv:1903.00784. Mastering Diverse Domains through World
  Models (DreamerV3, 2023), arXiv:2301.04104.
- Unsupervised Predictive Memory in a Goal-Directed Agent (MERLIN, 2018), arXiv:1803.10760. Memory Gym (2023),
  arXiv:2309.17207.
- Asymmetric Actor Critic for Image-Based Robot Learning (2017), arXiv:1710.06542.
- Learning Quadrupedal Locomotion over Challenging Terrain (2020), arXiv:2010.11251. RMA: Rapid Motor Adaptation for
  Legged Robots (2021), arXiv:2107.04034. Learning robust perceptive locomotion for quadrupedal robots in the wild
  (2022), arXiv:2201.08117. Legged Locomotion in Challenging Terrains using Egocentric Vision (2022),
  arXiv:2211.07638. Neural Volumetric Memory for Visual Locomotion Control (2023), arXiv:2304.01201. Learning to Walk
  in Minutes Using Massively Parallel Deep RL (2021), arXiv:2109.11978.
- Learning To Explore Using Active Neural SLAM (2020), arXiv:2004.05155. Cognitive Mapping and Planning for Visual
  Navigation (2017), arXiv:1702.03920. Neural Map (2017), arXiv:1702.08360. Hydra: A Real-time Spatial Perception
  System for 3D Scene Graph Construction (2022), arXiv:2201.13360.
- Simple Online and Realtime Tracking (2016), arXiv:1602.00763. Learning to Track with Object Permanence (2021),
  arXiv:2103.14258.
- Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains (2020), arXiv:2006.10739.
- Machine Theory of Mind (ToMnet, 2018), arXiv:1802.07740. Agent Modelling under Partial Observability (LIAM, 2021),
  arXiv:2006.09447.
- Hegde et al. (2021), agents hearing sound in ViZDoom, arXiv:2107.02195 (the identifier was confirmed by search; the
  title was not).
- RT-1 (2022), arXiv:2212.06817. RT-2 (2023), arXiv:2307.15818. OpenVLA (2024), arXiv:2406.09246. Octo (2024),
  arXiv:2405.12213. pi0 (2024), arXiv:2410.24164.
- Lee, D. N. (1976). A theory of visual control of braking based on information about time-to-collision.
  Perception 5, 437.
