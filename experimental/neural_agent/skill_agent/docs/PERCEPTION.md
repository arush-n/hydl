# The skill layer's perception: an open content schema (design, 2026-10-19)

**Status: accepted by the owner on 2026-10-19, with the decisions of section 8. Build steps 1 and 2 are done: the
registry (`perception/registry.py`) holds the items, blocks, gather types, creatures, attitudes and description
vectors; the v2 token types stand beside v1 (`contract/surface.py`, section 4 "As built") and the trunk reads them;
the v2 adapter comes next.** Owner rulings that frame it (2026-10-19): the
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

**As built** (step 2, 2026-10-19; `contract/surface.py`, schema `skill_tokens_v2`): entity 89 features, block
candidate 71, inventory slot 74, self 36; geometry (14) and kit (15) keep v1's. Each content token holds its v1 state,
the registry's property row (pinned in the contract: 14 creature, 15 block and 21 item properties, checked against
the registry by a test), a 32-number description vector, 8 reserve features, `known` (its id is in the registry) and
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

**Asks of the simulator layer** (a contract request; that code is not this lane's): block candidates published with
their block id; each entity's role id; an entity's held item id when a player could see it; per-cell fluid ids in
geometry. The adapter maps each id to properties itself. An attitude changed at run time (the engine's override, and
friendship between NPCs spawned together) is state, not content; whether the observation publishes it is settled with
the adapter.

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
   width (proposal: 32) is set by the latency measurement of step 5.
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
