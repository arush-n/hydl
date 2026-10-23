# Asks of the simulator layer: what the skill layer's perception needs published (contract request, 2026-10-19)

**From** the skill layer (`experimental/neural_agent/skill_agent/`). **To** the owners of the learner observation
(`HytaleRL/hytalegym/hytalegym/jax/combat/observation/`) and of the gen-7 simulator. This is build step 4 of
`PERCEPTION.md`; nothing here changes their code.

**Why.** The skill layer's v2 tokens (`PERCEPTION.md` section 4, built in `contract/surface.py`) describe content by
properties and relations looked up in the content registry (`perception/registry.py`) by id. The learner observation
publishes few ids today, so most v2 content stays unknown: until an ask lands, the adapter fills what is published
and leaves the token's `known` feature at 0.

**Ids.** Every id below is `semantic_id(asset_id)`, the 31-bit FNV-1a of the asset's string id
(`jax/combat/contracts/semantic.py`), the scheme inventory item ids already use; 0 means none or unknown. Over the
0.7.2 assets no two ids collide (3,960 items, 3,182 blocks, 894 NPC roles; checked 2026-10-19).

## The asks, most valuable first

| # | Ask | Today | Proposed field | Legal basis |
|---|---|---|---|---|
| 1 | Which creature each nearby entity is | `entity_kind` is `ENTITY_KIND_NPC` (1) for every row (`observation/v1/schema/contract.py:246`, filled at `v1/runtime/encoder.py:140`) | `entity_i32` gains `role_id` (the NPC role's semantic id; 0 for a player) and a player kind | A player sees which creature it is (its model) |
| 2 | The agent's own inventory | `LearnerCombatObservationV3` has no inventory field (`v3/schema/types.py:27-62`), though the v3 inventory tokens exist (`v3/tokens/inventory.py:34-43`: container, slot, item id, quantity, durability, active) | Publish those tokens in the learner observation | The agent knows its inventory |
| 3 | More entities | `NEARBY_ENTITY_CAPACITY = 16` (`v1/schema/contract.py:16`), and the two-entity path fills slot 0 only (`v1/runtime/encoder.py:146-147`) | The capacity as a configuration, every perceptible entity filled (owner, 2026-10-19: 16 by default, 32 and above supported) | Each row stays gated by perception, as today |
| 4 | Blocks the agent can act on | No block token in the observation | A block-candidate set within reach: cell, face, the block's semantic id; capacity a configuration | Visible blocks in reach, gated by line of sight like entity rows |
| 5 | What an entity holds | Entity rows carry no held item (`ENTITY_INTEGER_FEATURES`: id, kind, attack phase, attack index; `v1/schema/contract.py:135-140`) | `entity_i32` gains `held_item_id` (0 when empty or not perceptible) | A held item is visible; the game's own rules decide, checked on the live 0.7.2 server (`PERCEPTION.md` decision 4) |
| 6 | Fluids | World geometry tokens carry semantic flags, no fluid id (`v3/tokens/world.py:34-50`) | The fluid's semantic id per geometry token | A fluid is visible |
| 7 | The agent's held item, by id | `weapon_i32` publishes `weapon_id`, `weapon_family`, `active_ability_slot` (`v3/schema/contract.py:20-24`) | Say whether `weapon_id` is the held item's semantic id; if not, publish that id | The agent knows what it holds |

### Asks 9 and 10, and an arena: cover at range (2026-10-20, for the guerrilla proof)

The owner's principle for these: "the more legal information the better, not every neural layer will activate each
input but if needed it should". Today the agent sees geometry only in its 9-by-9-block local window: the navigation
tokens reach 4 blocks, and `combat_cover` (9046a78) puts its pillars at radius 3 because the engine window ends
there. Hiding needs cover at the range a player picks it, 5 to 16 blocks.

| # | Ask | Today | Proposed field | Legal basis |
|---|---|---|---|---|
| 9 | The solid structures the agent sees at range | Nothing beyond the local window | Occluder tokens: the K nearest solid columns tall enough to hide a standing agent (2 blocks or more) whose face the agent can see within the sensor range, as (cell x, z relative to the agent, base y, height, block semantic id); K a configuration, 16 by default | The face is in line of sight and in the view sector, the same gate as an entity row |
| 10 | Where to stand behind them against a threat | Nothing | For each occluder token, the cell beside it on the side away from the target, with a flag that the column blocks the target's line of sight to a standing agent there, the walking distance, and whether a path exists | Derived from ask 9's visible geometry and the target's position as the agent knows it (seen, or last known), a computation a player makes by eye |

**The arena that tests them.** A guerrilla arena on the Region geometry route (`RegionGeometryState` with a region
fixture), so cover can sit 5 to 16 blocks out:
- pillars and walls scattered around the spawn;
- the agent with a ranged loadout once the bow fires;
- an active melee opponent;
- the legal sector and the sensor range.

Ask 10 is the convenience form of ask 9. If only one lands, ask 9 is the input a network needs; the cover point can
then be learned, or computed in the skill layer from ask 9.

The arena is also what ranged combat needs. On the open-flat and local-window fixtures a target beyond the 9-block
window freezes the arsenal: `geometry_exhausted` is set, abilities never tick, and nothing reports it (the
coordinator's bow recipe, 2026-10-20). A shot fires at 3.5 blocks and freezes at 4.5. A bow there works only at sword
range; at range it needs the Region route.

### Asks 11 to 16: what the encoder audit found missing (2026-10-21)

From the field-by-field audit of the learner observation and the literature check (`PERCEPTION.md` section 11). Sent
to the coordinator on 2026-10-21. S16 is filed under ask 3, which the owner assigned; S11 to S15 are new scope,
waiting on the owner. S17 sent the same day. Since then: S13 APPROVED 2026-10-22 night ("measure live first"); S12
and S14 APPROVED 2026-10-23 (the owner, on the recommendation to start with them: "ok continue in parallel"); S11
and S15 still wait.

| # | Ask | Today | Proposed field | Legal basis |
|---|---|---|---|---|
| 11 | Damage and hits, per actor (high) | `NativeDamageEventBatch` (`v3/native/channels/events.py`) exists for native traces only; the learner sees its health fraction and the `damage_recovery` timer, with no direction or source | Recent events at a fixed capacity. Received: amount, cause and class, blocked, lethal, the source's direction in the actor's frame, and the source entity id only when it is perceivable. Dealt (hit confirm): target id, amount. The age of each | A player sees the damage indicator, even for an attacker behind it, and a hit confirmed on its target |
| 12 | Sound (medium-high; part of S9) | `jax/world/audio` (`AudioObservation`: source position, category, block walk/land/hit/break/build and item-drop families, maximum distance, volume) has no consumer | The sounds an actor hears, attenuated by their maximum distance: source position (or bearing and distance), category, family and ordinal, volume, age | A player hears footsteps and block breaks of what it cannot see |
| 13 | Item drops (medium) | No item-entity or pickup rows in the observation | Nearby dropped items: position, item id, count | A dropped item is visible |
| 14 | Clock and weather (medium) | The light channel is a surrogate sky light computed from terrain height; no time of day | Time of day, time to the next day or night, weather | A player sees the sky |
| 15 | An entity's team or group (medium; multi-agent) | Entity rows carry only hostile and attitude | Each perceived entity's group or faction id, relative to the actor | A player sees a teammate's marker or colours |
| 16 | One velocity unit for entity and self rows (medium) | The self row's velocity is over `agent_maximum_gait_speed`, an entity row's relative velocity over one target's `target_chase_speed` (`v1/runtime/encoder.py`) | Both in one unit, or the scale per row | Unchanged; without it the stored memory cannot carry an entity's own velocity across ticks |
| 17 | My own maximum health; a block's break progress (low) | The self row has the health fraction only; `ActorBlockActionParameters.block_health` exists but is not in the block candidate's policy view | `self_f32` gains the maximum health; the block candidate gains its health fraction | A player sees its own health numbers and a block's cracks |

## Status

**2026-10-19: all seven accepted** by the 0.7.2 port's coordinator (the simulator session), scheduled as the next
simulator task after the 0.7.2 certification unless the owner reorders. The shape: an opt-in gen-7 perception channel
(working name `perception_v2`) produced by the render beside the v3 observation, behind the gen-7 profile switch, so
the v1 and v3 layouts and the gen-5 identity fingerprints stay bit-identical and nothing in
`jax/combat/observation` changes in place. Every per-entity and per-block field passes the legal-observation gate of
the certified pair path (view sector, sensor range, line of sight with the detail-hitbox rule); a role or held item
is never published for an entity row the gate blanks. The channel's contract (layout, capacities, gate rules) comes
before any field lands.

| # | Status | Field, as proposed by the simulator side |
|---|---|---|
| 1 | channel built (2026-10-19 night); the role id is a seam | `entity_kind` filled (1 NPC, 2 player; read as `player`); `entity_role_id` is 0 until the runtime keeps role ids (the seam in `PERCEPTION_V2.md` section 6), so `known` waits on it. The adapter resolves a role the registry holds (unit test with a synthetic channel). |
| 2 | channel built and read (2026-10-19 night) | `inventory_i32`, `inventory_f32`, `inventory_mask`: the adapter's `inventory_slot` tokens. |
| 3 | channel built as a configuration | `entity_capacity` (16 by default, any C >= 1); rows 1 and up stay 0 until a multi-entity encoder (seam). |
| 4 | reserved (seam) | `block_candidate_i32` and `block_candidate_capacity`: the engine inputs are named in `PERCEPTION_V2.md` section 6. |
| 5 | channel built and read (2026-10-19 night) | `entity_held_item_id`. A target without a weapon published -1 in the open-flat arena (its loadout's sentinel); reported, and the coordinator fixed the channel to 0 (with the channel's commit). The adapter reads any id of 0 or below as nothing held. |
| 6 | reserved (seam) | `fluid_id` per geometry token. |
| 7 | read (2026-10-19 night) | The agent's row of `weapon_i32` (`weapon_id`) fills the self token's `holds` and the held item's registry row. |
| 8 | channel built and read (2026-10-19 night) | `body_yaw_sin`, `body_yaw_cos`: the adapter takes the body yaw from them when the caller passes none. |

**2026-10-19 night: the channel is built** (`HytaleRL/hytalegym/hytalegym/sim/gen7/PERCEPTION_V2.md`,
`sim/gen7/mechanics/perception_v2.py`), opt-in and off in both profiles; `runtime/gen7.py --construct perception_v2`
switches it on in one process. The gate is the entity row's own (`base.entity_mask`): line of sight, the 24-block
radius, and the view sector and sensor range only where the arena passes `view_sector_full_angle_radians` and
`legal_observation_sensor_range=True` to `geometry_arsenal_world_capabilities`. The composition's open-flat arena
passed neither, so an agent there perceived a target behind it. Since ff7ef22 it passes the sensor range, which takes
effect (a target 20 blocks ahead is blanked), and a 120-degree sector set on the returned capabilities, which does not
reach the row gate yet: a target dead behind the agent, or 90 degrees to its side, is still published (probe at
9159cfd; reported to the coordinator).

## Already published, and read by the v2 adapter

Each entity row's position and velocity in the agent's frame, visibility, health fraction and `alive`; its world yaw
as sine and cosine (`v1/runtime/encoder.py:445-446`; the adapter turns it into the body frame with the agent's yaw);
its attack phase (the argmax of the five `target_phase_*` columns, in `entity_i32`) and attack progress. The self row
and the resource meters, as the v1 adapter reads them.

## What the skill layer tests when an ask lands

The three-world leakage test over the new field (an entity hidden behind a wall, or none, must not change it); the
adapter's `known` feature turning on; a registry lookup of the published id giving the asset's properties (for
example, an iron pickaxe in the hotbar gives power 0.25 against iron ore).

Results (2026-10-19 night, `test_sa_perception_v2.py`; a private gen-7 render with the channel on, the gym's wall
geometry, a 120-degree view sector and the sensor range on): with the target in the open the row is published (kind
1, held 0, read as nothing held); behind a wall, absent, and behind the agent outside the sector, kind, held item and
role are 0 in every row and the adapter's entity tokens are all zeros. `known`: blocked on the role seam. The engine's
own hotbar sword is published as `semantic_id("Weapon_Sword_Iron")` (so the inventory's item ids are the assets'
semantic ids, which `PERCEPTION_V2.md` section 4 had inferred) and resolves as a known weapon; an iron pickaxe placed in
hotbar slot 1 is published as `semantic_id("Tool_Pickaxe_Iron")` and gives power 0.25 against `Ore_Iron_Basalt`.
