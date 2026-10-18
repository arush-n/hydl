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

## Answers (2026-10-18; FIDELITY-0.7.2.md section (s); jar `A6C310DD`)

- Stage 1: not a plain click-craft. The structural craft queues from the window's one-slot input with ORDERED removal and
  never calls `craftItem`; the bridge moves the inputs into that slot at the request tick and the job lands one tick later.
- Stage 2: the tier is `BenchBlock.tierLevel`; a fresh bench is at 1; tiers 1 to 8 are reachable natively; the fixture sets
  it with `setTierLevel`, the readout is `native_crafting_bench_tier`.
- Stage 4: the level is not fixed. It comes from the universe-wide recorded-memories count (`MemoriesPlugin`), set by the
  `memories_level` reset option; the readout is `native_memories_level` and `native_memories_recorded_count`.
- Oracle route: reset-time probes (`crafting_oracle_requests`), scope `oracle_unbound_*`; the production step path still
  requires candidate evidence for bench contexts until the policy capture offers them (cutover).

## Answers, stages 3 and 5 (2026-10-18; FIDELITY-0.7.2.md section (t); jar `5FE609E2`)

- Stage 3: a timed bench craft is a multi-tick native job. Inputs leave the inventory on the first world tick after the
  request; the output lands when the native progress reaches `TimeSeconds` (3.0 s = 87 world ticks at 1/30 s; model it on
  progress seconds, not a fixed tick count). Cancels: walking out of reach (7 blocks, window open) cancels with a refund to
  the original slots; a second craft request queues behind and does not cancel; taking damage does not cancel; an ability
  request has no cancel path (inferred, not run). Fieldcraft has no timed recipe and crafts instantly. Probe shape:
  `crafting_oracle_requests[i]` gains `open_window` and `events` (`tick`, `kind` reach_out | second_craft | damage).
- Stage 5: processing benches do not use `craft_recipe`: items go into the bench's input and fuel containers, the bench
  activates (`SetActiveAction`), and the processing tick advances on game time; the route is container placement plus time.
  Not proven live: the bench did not advance in the bridge world (its game clock is paused). The Armory is a click-craft
  (`CraftItemAction` -> `queueCraft` ORDERED), not `craft_recipe`; no separate diagram item is needed (reader's read); not
  run live. Both stay open on the bridge side.

## Answers, stage 5 measured (2026-10-18; FIDELITY-0.7.2.md section (v); jar `4A5E5D6D`)

- Processing: container placement plus game time, with the world and the game clock running (the bridge's stepped world
  pauses both, which is why nothing advanced before). Campfire fish (2.0 s): output 61.9 game seconds after activation, 58 to
  59 world ticks; one stick (4.0 s of fuel) consumed 2 ticks after activation; the input consumed at the output. Furnace copper
  bar (10 s): output at about 301 game seconds, 286 to 289 world ticks; three sticks; one charcoal at the second fuel consumption
  (the asset's extra output, one per two fuel items, inferred from the asset plus one observation). Model on game seconds
  (about 1.0 to 1.07 game seconds per world tick across launches), not tick counts. Probe: `processing_oracle_requests`.
- Diagram (Armory): a click-craft through `CraftItemAction` with ORDERED removal from the window's positional input slots; the
  Weapons/Axe category must be selected first (a window setting, not an item; the default is Weapons/Sword); no diagram item
  beyond the recipe's inputs; queued at the request tick, inputs out and the axe in two ticks later; recipe time 0.0 s.
  Probe: `crafting_oracle_requests` with `crafting_context` "diagram", `category`, `item_category`.
- Open: Tannery and Salvagebench, processing refusals, diagram negative controls; the simulator needs a processing action
  (insert, activate, advance by game seconds) and a diagram action (category, positional inputs, craft) rather than
  `craft_recipe`.
