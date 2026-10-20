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
| 1 | accepted, scheduled after certification (2026-10-19) | `entity_role_id` (i32; 0 for a player); `entity_kind` gains `ENTITY_KIND_PLAYER` |
| 2 | accepted, scheduled after certification (2026-10-19) | `inventory_i32`, `inventory_f32`: the v3 inventory tokens republished unchanged |
| 3 | accepted, scheduled after certification (2026-10-19) | `entity_capacity`, a configuration (16 by default, 32 and above), rows padded with id 0 |
| 4 | accepted, scheduled after certification (2026-10-19) | `block_candidate_i32` rows (cell x, y, z, face, block_id, reachable) and `block_candidate_capacity`, mirroring the bridge's native world-action capture |
| 5 | accepted, scheduled after certification (2026-10-19) | `entity_held_item_id` (i32; 0 when none or not legally observed) |
| 6 | accepted, scheduled after certification (2026-10-19) | `fluid_id` per geometry token (i32; visible fluids only) |
| 7 | answered (2026-10-19) | `weapon_id` is already the held weapon's semantic id (`jax/combat/arsenal/profiles/hytale_0_5_7.py:989`, checked); a held item that is not a weapon comes through `entity_held_item_id`. The adapter can read the held weapon now (a follow-up). |

## Already published, and read by the v2 adapter

Each entity row's position and velocity in the agent's frame, visibility, health fraction and `alive`; its world yaw
as sine and cosine (`v1/runtime/encoder.py:445-446`; the adapter turns it into the body frame with the agent's yaw);
its attack phase (the argmax of the five `target_phase_*` columns, in `entity_i32`) and attack progress. The self row
and the resource meters, as the v1 adapter reads them.

## What the skill layer tests when an ask lands

The three-world leakage test over the new field (an entity hidden behind a wall, or none, must not change it); the
adapter's `known` feature turning on; a registry lookup of the published id giving the asset's properties (for
example, an iron pickaxe in the hotbar gives power 0.25 against iron ore).
