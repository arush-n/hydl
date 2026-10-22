# PERCEPTION_V2: the gen-7 perception channel (contract)

Lane: task #92 (0.7.2 port, perception asks of `experimental/neural_agent/skill_agent/docs/PERCEPTION_ASKS.md`).
Status: contract (step A, 2026-10-19). The implementation is `sim/gen7/mechanics/perception_v2.py` (construct) and
`tests/sim/test_perception_v2.py`; section 9 records what is built. Nothing in `hytalegym/jax` changes: the channel is
produced by the render (hook row `perception_v2` in `sim/gen7/overlay.py`).

## 1. What the channel is

- **Opt-in.** `Constructs.perception_v2: bool = False` (`sim/common/profile.py`). Gen 5 and the gen-7 profile keep
  `False`, so every existing output is bit-identical. Flipping the gen-7 profile to `True` is the coordinator's call (it
  also needs a `mechanics_072.md` row).
- **Beside the v3 observation.** When the switch is on, `encode_learner_observation_v3` (the render's hook) returns
  `LearnerCombatObservationV3WithPerceptionV2`: every v3 field, unchanged in value and order, plus one field
  `perception_v2`, a `PerceptionV2` NamedTuple of the arrays below. When off, the v3 type is returned as today.
- **Legal by construction.** Entity fields are blanked by the entity row's own gate (section 3), so a field is never
  published for a row the observation does not publish. The channel adds no perception of its own.

## 2. Layout

B = batch; C = entity capacity (default 16); S = inventory token capacity, 76 (`observation/v3/tokens/inventory.py`
`INVENTORY_POLICY_TOKEN_CAPACITY` = 36+4+9+4+23+0 over six containers). Padding is id 0 (or False / 0.0).

| Array | dtype | shape | Meaning | Today | Gate |
|---|---|---|---|---|---|
| `entity_kind` | int32 | [B, C] | 0 none, `ENTITY_KIND_NPC` = 1 (as v1), `ENTITY_KIND_PLAYER` = 2 (new, channel-local) | filled, row 0 | row gate |
| `entity_role_id` | int32 | [B, C] | semantic id of the target's NPC role asset | filled, row 0 (0 for a player-backed target or an empty role; section 6) | row gate |
| `entity_held_item_id` | int32 | [B, C] | semantic id of the target's held weapon | filled, row 0 | row gate |
| `inventory_i32` | int32 | [B, S, 3] | (container_id, container_slot, item_id) per token; 0 where absent | filled | actor validity |
| `inventory_f32` | float32 | [B, S, 7] | the v3 inventory token features, unchanged | filled | actor validity |
| `inventory_mask` | bool | [B, S] | token present | filled | actor validity |
| `body_yaw_sin`, `body_yaw_cos` | float32 | [B] | sin and cos of the body yaw `state.combat.yaw[:, AGENT]` (degrees, converted to radians); forward = (-sin yaw, 0, -cos yaw) | filled | none (self) |
| `self_position` | float32 | [B, 3] | the agent's own world-frame position (x, y, z in blocks), `state.combat.position[:, AGENT]`. Appended last, after `body_yaw_cos` (task #96, 2026-10-19; ruling in section 7) | filled | none (self) |

Row 0 of the entity arrays is the target. The two-entity path publishes one entity row (the v1 encoder writes
slot 0 only; `combat_scene_from_state` and `encode_learner_observation`), so rows 1..C-1 are 0 until a multi-entity
encoder fills them (seam). `entity_capacity` as a configuration: any C >= 1 is accepted by the channel builder; the
render uses 16; 32 and above are tested by shape.

Reserved, not built (seams, section 6): `block_candidate_i32` [B, K, 6] = (cell x, y, z, face, block_id, reachable)
with `block_candidate_capacity` K; `fluid_id` per geometry token.

Weapon id: no new field. `weapon_i32`'s `weapon_id` already holds the semantic id of the held weapon asset
(`arsenal/profiles/hytale_0_5_7.py:988-989`: `semantic_id(profile["asset"])`), so the agent's own weapon is already covered.

## 3. Gate per field (what the entity fields reuse)

The entity fields are published for row 0 only where the observation publishes that row. The gate is the entity row's
own mask, and the channel uses nothing else:

1. **Target perception.** `perceptible = combat_observation[:, TARGET_VISIBLE] > 0.5`
   (`observation/v1/runtime/encoder.py:409`, `_target_observation_evidence`).
2. **Row mask.** `entity_mask = target_mask & (distance <= NEARBY_ENTITY_RADIUS_BLOCKS)`, where `NEARBY_ENTITY_RADIUS_BLOCKS`
   is 24 (`observation/v1/runtime/encoder.py:113-115`, `combat_scene_from_state`; `base.entity_mask` of the v3 observation).
3. **Line of sight from the world capabilities.** `observation/v3/encoding/encoder.py:187-190`:
   `perception_line_of_sight = (world_capabilities.line_of_sight & world_capabilities.line_of_sight_valid)[:, AGENT]`.
4. **Opt-in view sector and sensor range.** Both reach the entity row through `line_of_sight`, not through a separate
   mask: `line_of_sight_value[:, AGENT] &= pair_agent_los_gate` (`arsenal/environment.py:2994-2997`), with the engagement
   gate at `:2568-2608` (`view_sector_full_angle_radians` and `legal_observation_sensor_range` of
   `geometry_arsenal_world_capabilities`, `:2423`). A default arena passes neither flag, so its row is line-of-sight only.
5. **Line of sight itself.** The gen-7 detail-hitbox rule (render hook 2b, `sim/gen7/mechanics/line_of_sight.py`) through
   `world/geometry`'s `line_of_sight_result`.

Channel rule: `field = where(entity_mask[:, 0], value, 0)` for `entity_kind`, `entity_held_item_id` and (when it lands)
`entity_role_id`. The gate does not depend on `ArsenalRuntimeConfig.legal_observation`: a row the world capabilities
blank carries no role or held item whatever that flag says.

Self fields (`inventory_*`, `body_yaw_*`) are the agent's own state and carry no perception gate; the inventory rows are
gated by actor validity (`observation.valid`), as the v3 carry already does (`arsenal/environment.py:1616`).

## 4. Id scheme

- Every id is `semantic_id(asset_id)` (`jax/combat/contracts/semantic.py:6`): the positive 31-bit FNV-1a of the asset string.
- 0 means none, not held, absent, or unknown. The channel never writes a non-semantic id.
- `entity_role_id` is `semantic_id(role name)` for the target's role, read from `ArsenalRuntimeConfig.entity_role_semantic_ids`
  (`int32 [B, E]`, written by the construct `perception_v2:construct_roles` on `arsenal_runtime_config`, 2026-10-22). The names
  are the caller's `entity_role_ids` (the duel's `roles=`), or the matchup pair when none is passed
  (`default_combat_role_ids`: agent `Kweebec_Razorleaf`, target `Trork_Brawler`, `rulesets/hytale_0_5_7/*_v9.json`). An empty
  name is 0; a player-backed target is 0, since a player has no NPC role asset. The agent's own role is on the config (entity 0);
  the layout has no self role field, so the channel does not publish it.
- `entity_held_item_id` is `ArsenalRuntimeConfig.loadout.weapon_id[:, TARGET]` (`arsenal/schema/types.py:61-64`), which
  holds `semantic_id` of the weapon asset by construction (section 2).
- `inventory_i32[..., 2]` is the `InventoryState.item_id` of the agent's slot. INFERRED that this is `semantic_id` of the
  item asset: the stack contract types it `int32_semantic_id` (`inventory/schema/contract.py:79`) and the arsenal derives
  item ids the same way (`arsenal/environment.py:621`, `arsenal/world_actions/executor.py:306`). The test checks the channel
  passes the state's id through; it does not check the native-frame conversion.

## 5. Configuration

| Knob | Where | Default | Notes |
|---|---|---|---|
| `perception_v2` | `Constructs` (`sim/common/profile.py`) | `False` | gen 7 keeps `False` |
| `entity_capacity` C | `channel(..., entity_capacity=C)`; the render uses 16 | 16 | C >= 1; rows >= 1 zero until multi-entity |
| `block_candidate_capacity` K | seam | reserved | the bridge's capture capacity sets it (section 6) |

## 6. Seams (named, not built)

- **`entity_role_id` (closed, task #130 item 4, 2026-10-22).** `arsenal/runtime.py:194` takes `entity_role_ids` and feeds
  `initial_role_status_applications` (`:416`), which did not keep them, so the channel wrote 0 for every entity. The config now
  carries `entity_role_semantic_ids` (patch `perception_role_config_field`, `arsenal/schema/types.py`, None unless the switch is
  on) filled by the hook `perception_roles` (`construct_roles`), applied after the hit-interrupt role hook so it reads the raw
  names. The peer's `known` flag can turn on for a registry-resolved role; the registry lookup is on the peer's side.
- **Block candidates (ask 4).** The rows come from the world candidate producer behind
  `ArsenalActorBlockCandidateSource` (`arsenal/environment.py:309`; provider `:960`) into `ArsenalActionSurfaceEvidence`
  (`:322`). They are not encoder inputs, and the world tokens carry no cell or block id (`observation/v3/tokens/world.py:94-99`).
  Seam: pass the candidate rows (cell, face, block id, reachable) from the action surface at the step into the channel builder.
- **Fluid id (ask 6).** The geometry carries fluid levels and movement per cell (`jax/world/geometry/__init__.py:78-95`:
  `fluid_level`, `fluid_fill_height`, `fluid_damage`, `fluid_movement`) and no fluid asset id. Seam: a per-cell fluid id
  in the capture, then `fluid_id` per geometry token with the same visibility gate as the geometry it describes.
- **Multi-entity rows.** Seam: the encoder fills row 0 only; rows >= 1 need a multi-entity encoder.

## 7. Contract for the peer's adapter

- `observation.perception_v2` exists only when the gen-7 render runs with the switch on (the observation then has the
  type `LearnerCombatObservationV3WithPerceptionV2`). With the switch off, the observation is the v3 type, unchanged.
- Entity ids: 0 for padded rows and for any row the gate blanks. A non-zero id on a blanked row is a leak.
- Kinds: 0 none, 1 NPC, 2 PLAYER. A player kind on a row means the target is player-backed (`player_backed_actor_mask`).
- The inventory and body yaw arrays are the same values the v3 inventory tokens and the self yaw carry.
- `self_position` (task #96, 2026-10-19) is the agent's own position, the value of `state.combat.position[:, AGENT]`. Ruling
  (coordinator, 2026-10-19): a player reads their own coordinates, so the field is legal. It is gen-7 only, like the rest of
  the channel, and it is appended after `body_yaw_cos`, so no existing field moves (append-only).

## 8. Test plan

Ours (`tests/sim/test_perception_v2.py`):
- shape and dtype of every array, for C = 16 and C = 32;
- the switch: off returns the v3 type with unchanged fields; on returns the augmented type with the v3 fields unchanged;
- the row gate: a target outside the view sector, and a target behind an occluder (through the rendered line-of-sight
  construct), both blank `entity_kind`, `entity_held_item_id` and `entity_role_id`; a visible target fills them;
- task #130 item 4 (2026-10-22, on a render with the switch on, `on_engine`): the switch-off config carries no role ids; the
  matchup default gives the target row `semantic_id("Trork_Brawler")`; the duel's `roles=` reaches the config and the row; swapped
  roles put `Kweebec_Razorleaf` on the target row; an empty role name reads 0; a player-backed target has kind PLAYER and role 0;
- ids: `entity_held_item_id` equals `semantic_id(weapon asset)` for the target's loadout;
- body yaw equals sin and cos of the body yaw in radians;
- inventory tokens equal `inventory_policy_tokens_from_state`'s floats;
- defaults bit-identical: `test_render_identity.py` and the rollout fingerprints `d8477b63...` and `b02ee274...`.
- task #96 (2026-10-19): `self_position` equals `state.combat.position[:, AGENT]` for a non-origin agent; the field order is
  the eight older fields, then `self_position`; gen 5 keeps its construct absent (`construct(original, GEN5) is original`).

Peer's (from `PERCEPTION_ASKS.md`):
- the three-world leakage test over each field (target visible; target hidden behind a wall; no target), each filled field
  zero in the hidden and empty worlds;
- the adapter's `known` flag turning on for a registry-resolved role (needs the role seam);
- a registry lookup of the published id returning the asset's properties (for example, an iron pickaxe in the hotbar gives
  power 0.25 against iron ore).

## 9. Status

Built (lane task #92): `sim/gen7/mechanics/perception_v2.py` (construct, `channel()`, `observation_type()`; the inventory
source replicated from `observation/v3/tokens/inventory.py:142-185` with the same arguments). Tests:
`tests/sim/test_perception_v2.py`, 15 cases (numbers below). Shared edits (after the overlay lock): `Constructs.perception_v2`
(`sim/common/profile.py`) and the hook row `perception_v2` in `sim/gen7/overlay.py`.

Coverage rule (the channel's rule, not a defect). The channel follows the entity-row gate exactly. The row publishes the
target only where `observation.base.entity_mask` is true: the target is perceptible through the agent's line of sight,
`world_capabilities.line_of_sight & line_of_sight_valid` (`observation/v3/encoding/encoder.py:187-190`), and within
24 blocks. The view sector and the sensor range reach that line of sight only when the arena passes
`view_sector_full_angle_radians` (finite, below tau) and `legal_observation_sensor_range=True` (two-entity path) to
`geometry_arsenal_world_capabilities` (`arsenal/environment.py:2423`; engagement gate `:2568-2608`; folded into
`line_of_sight` at `:2994-2997`). Otherwise the row is line-of-sight only (the gen-7 detail-hitbox rule). `legal_observation`
does not enter this gate. So the peer's legality requirement holds in arenas that set both flags, and the peer must set
both: `view_sector_full_angle_radians` and `legal_observation_sensor_range=True`.

Seams, with the engine input each needs (not built):
- Role id (`entity_role_id`): closed 2026-10-22 (section 6, the first bullet; section 9).
- Block candidates (ask 4): the candidate rows (cell x, y, z, face, block_id, reachable) from the action surface at the step,
  `ArsenalActionSurfaceEvidence` (`arsenal/environment.py:322`), built by `ArsenalActorBlockCandidateSource` (`:309`) and
  `make_world_actor_block_candidate_provider` (`:960`). Input: those rows passed to the channel builder beside the
  observation, and a `block_candidate_capacity`. The world geometry tokens carry no cell or block id
  (`observation/v3/tokens/world.py:94-99`).
- Fluid id (ask 6): a per-cell fluid asset id in the geometry capture. `jax/world/geometry/__init__.py:78-95` carries
  `fluid_level`, `fluid_fill_height`, `fluid_damage` and `fluid_movement`, and no asset id. Input: the fluid block id per
  cell, with the same visibility gate as the cell.
- Multi-entity rows: the encoder fills row 0 only; rows 1 and up need a multi-entity encoder.

Results (2026-10-19, after the shared edits; logs in tmp/perception_v2/after/ and tmp/perception_v2/baseline/):
- tests/sim/test_perception_v2.py: 15 of 15 passed (focused run and full run).
- tests/sim/test_render.py: 22 of 22 passed. tests/sim/test_constructs.py: 15 of 15 passed (both runs).
- Rollout fingerprints after: batch 8 d8477b63fac9d8c550d35158a823c6324267d4de1801bbde7d1f92739dd4bd4b; batch 64
  b02ee27417fb649a44ea2e68a4632453816913af2c85078471f2d0f89924d8e7. Both equal the before values: the default path is
  bit-identical with the switch off.
- Identity gate (tests/sim/test_render_identity.py): 4 of 4 passed in the full run.
- tests/sim, full run. Before (20:30 tree): 324 passed, 0 failed, EXIT 0. After: 356 passed, 5 failed, 361 collected, EXIT 1.
  By file: the 28 files unchanged since 20:30 pass exactly as before (288 tests). test_render.py and test_constructs.py
  together are 37 (36 before; another lane changed both after 20:30). New files: test_perception_v2.py 15 of 15 (this lane);
  test_hit_interrupt_statuses.py 14 of 14 and test_crafting_augment_tags.py 2 passed, 5 failed (other lanes). The five
  failures name symbols the tree lacks for that lane (Constructs.crafting_augment_tags, bench_augment_mask,
  bench_augment_tags); none touches perception_v2.

Task #96 (2026-10-19): `self_position` (float32 [B, 3], the agent's `state.combat.position[:, AGENT]`) is the last field of
`PerceptionV2`, after `body_yaw_cos`. Ruling by the coordinator, 2026-10-19: a player reads their own coordinates, so the
field is legal; gen 7 only, like the channel; append-only. Results (2026-10-19, same tree before and after the change unless
noted):
- `tests/sim/test_perception_v2.py`: 19 of 19 passed (after). Three cases are new: the self position equals the combat
  state's agent position, the field order, and gen 5's construct staying absent.
- `tests/sim`, full run (`JAX_PLATFORMS=cpu`, `-p no:cacheprovider`): before 3 failed, 360 passed, 1 skipped, exit 1; after
  3 failed, 363 passed, 1 skipped, exit 1. The three failures are the same in both runs, all in
  `test_crafting_augment_tags.py` (another lane); the +3 are the new cases.
- Identity gate, `tests/sim/test_render_identity.py`, run on its own: before 4 passed (exit 0); after 4 passed (exit 0).
- Rollout fingerprints (`tools/sim_bench/rollout.py`, batches 8 and 64, 64 steps): before `d8477b63…bd4b` and
  `b02ee274…d8e7`; after the same two values. Unchanged.

Task #130 item 4 (2026-10-22): the role id. Cause (confirmed, `perception_v2.py` before this change): the channel wrote a zero
array for `entity_role_id`, and `arsenal_runtime_config` did not keep the role names it was given. Fix: the config field and
the hook above; the channel publishes `where(present & ~player, config.entity_role_semantic_ids[:, TARGET], 0)`. Evidence:
`tests/sim/test_perception_v2.py` on a gen-7 render with the switch on (the matchup default reads `semantic_id("Trork_Brawler")`,
the duel's roles and swapped roles read as named). Red check: with the channel's role read zeroed, the three value tests fail and
the empty-role and player-backed tests pass, as they should. Not done: a self role field (the layout is peer-facing; the agent's
role is on the config only); the peer's `known` flag and attitude are the peer's to derive.
