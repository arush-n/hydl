# Crafting oracle contract for the 0.7.2 bridge (2026-10-18)

The simulator side (the fidelity session, `sim/gen7` crafting constructs; FIDELITY-0.7.2.md sections (o) and later)
has 263 of the 2,047 recipes of the 0.7.2 catalog committing after crafting stages 0 and 1. The remaining stages need
native evidence the bridge does not expose today. This file is the contract the bridge work is briefed from, written by
the fidelity session and kept here so the agents building the bridge side read the same text. Catalog counts are
bench-requirement rows from the 0.7.2 manifest (section (l)).

## Base record, every oracle request

All five stages extend the existing `craft_recipe` verb (`crafting_context`, `bench`, `expected_bench_block_id`,
`expected_bench_id`, `expected_bench_type`, `expected_bench_tier`). Every oracle returns, per request:

- accepted or refused, with the reject reason;
- the per-tick lifecycle flags;
- the inventory delta per container slot: inputs consumed and outputs added, with the tick each lands on;
- the bench readout the request resolved against: block id, bench id, type, tier.

## Stages, ordered by recipes gained per bridge effort

### 1. Structural (bench type 3, the Builders bench; 910 rows, mostly zero-time with concrete inputs)

- Fixture: one Builders bench block in reach.
- Requests: `Cloth_Block_Wool_Black_Half_Recipe_Generated_0` and `Cloth_Block_Wool_Black_Stairs_Recipe_Generated_0`
  with inputs stocked; one request with inputs missing.
- Question: is it a plain click-craft like type 0? If yes, the JAX side only needs the type-3 context.

### 2. Bench tiers (type 0, tiers 1 to 8; 166 rows, mostly Farmingbench, also Weapon_Bench, Armor_Bench and others)

- Fixture: place the bench at tier T. Needed: the native representation (which block state or component holds the
  tier) and how a fixture sets it.
- Readout: the in-reach bench's current tier, through block-candidate evidence or a perception key (JAX sets
  `bench_tier_level` 0 for every Region bench today; the tier per placed bench is needed).
- Requests: `Ingredient_Life_Essence_Lettuce_Recipe_Generated_0` (Farmingbench tier 2) at tier 1 (expect refused) and
  tier 2 (expect accepted); one tier-4 and one tier-8 recipe likewise (Cauliflower, Onion).

### 3. Timed recipes (time > 0; 734 rows, 82 blocked by time alone, e.g. `Armor_Copper_Chest_Recipe_Generated_0`, 3.0 s, Armor_Bench tier 0)

- Oracle, per tick from the request to completion: active, recipe id, elapsed or progress, finished, cancelled with a
  reason; the tick inputs are consumed (start or end) and the tick outputs appear; what cancels the craft (walking out
  of reach, a second craft request, taking damage, an ability request).
- Run at fieldcraft and at a bench, two seeds. The JAX side becomes a multi-tick craft operation.

### 4. Memories levels (2 to 6; 38 rows, all Furniture_Bench chests: Ancient 2, Desert 3, Temple_Emerald 4, Dungeon_Chest_Epic 5, Frozen_Castle 6)

- Readout: the learner's native memories level at reset, as an info key.
- Fixture: a reset option to set it.
- Requests: `Furniture_Ancient_Chest_Small_Recipe_Generated_0` at level 1 (expect refused) and level 2 (expect accepted).
- If the learner's level is fixed natively and cannot be set, report that; the JAX side then pins the level to the
  native value.

### 5. Processing (type 1: Salvagebench 421, Furnace 28, Tannery 8, Campfire 3) and Diagram (type 2: Armory 19, always paired with a Weapon_Bench tier)

- Processing is probably not a click-craft: inputs go into the bench container, often with fuel, and process over time.
  Oracle for the Campfire with `Food_Fish_Grilled_Recipe_Generated_0` (2.0 s) and for the Furnace: per tick, the
  container contents (input, fuel, output), fuel burn, output timing, and whether `craft_recipe` applies at all.
- Diagram: does `Weapon_Axe_Stone_Trork_Recipe_Generated_0` at an Armory need a diagram item, and is it a click-craft?

## Order

Stages 1, 2 and 4 first: their JAX side can be built and measured as soon as the oracle shape exists. Stages 3 and 5
need new mechanics on the simulator side after the oracle answers. The bridge refuses bench types 1 to 3 today
(`NativeHeadlessCrafting.java` 110-114).
