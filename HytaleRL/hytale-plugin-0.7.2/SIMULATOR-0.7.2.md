# Recalibrating the JAX simulator to Hytale 0.7.2 (plan)

Created 2026-10-17. Status: PLAN, nothing implemented. Owner decision
(2026-10-17): future agents train on 0.7.2 ("gen 7"); 0.5 is deprecated. The
bridge port is done ([`PORT-0.7.2.md`](PORT-0.7.2.md)); the simulator still
models 0.5.7, so every simulator checkpoint is labelled 0.5.7 and every
source/asset certification fails closed on 0.7.2
([`TRAINING-0.7.2.md`](TRAINING-0.7.2.md), "Not version-keyed on purpose").

Inventory below comes from three read-only surveys (2026-10-17) plus the measured
0.7.2 changes in `TRAINING-0.7.2.md`. Paths are relative to
`HytaleRL/hytalegym/hytalegym/` (`B/`) and `HytaleRL/hytalegym/tests/` (`T/`).

## Principles

1. **Version-keyed files, not edits in place** (owner rule for the port): each
   release line gets its own data files and profile modules, selected by
   `B/server_versions/` (`HYTALERL_SERVER_VERSION`). 0.5.7 stays bit-exact until
   it is retired, so existing evidence remains reproducible.
2. **Extract, then diff, then transcribe.** Most combat content is
   hand-transcribed from the 0.5.7 `Assets.zip` (no automated pipeline except item
   durability). Start by diffing 0.5.7 vs 0.7.2 sources so the work is the
   changed rows, not a rewrite.
3. **Every changed mechanic gets a native differential on a live 0.7.2 server**
   before it is called calibrated. A policy replay is not fidelity.
4. **Locomotion and world first.** The current skills (reach, pursue, escape) need
   movement, terrain, and line of sight; combat content can follow.

## Inputs on this machine

- 0.7.2: `~/Library/Application Support/Hytale/install/release/package/game/latest/`
  (`Assets.zip`, `Server/HytaleServer.jar`); 0.5: `.../install/v0.5/...`.
- Bridge `hytale-plugin-0.7.2/build/libs/HytaleRLBridge072-0.1.0.jar`, launcher
  `tools/launch_bridge_072.py`, smoke `tools/smoke_bridge_072.py`.
- Decompiled server sources: `/Users/arush/hydl/decompiled` (check which version;
  a 0.7.2 decompile is needed for engine semantics in phases 2-3).

## Phases and gates

### P0 Diff inventory (automated, no simulator change)
Script that opens both `Assets.zip` files and both server jars and reports, for
every asset the simulator transcribes (item interactions, `MovementConfig`,
`Stats/*`, NPC roles, `Mana.json`, crafting), which files and fields changed.
Seed list: `B/combat/assets/catalog.py:59-171` item lists, the provenance comments in
`B/jax/combat/arsenal/profiles/hytale_0_5_7.py` (5,083 lines; selectors 1976-2050),
`B/rulesets/hytale_0_5_7/*.json` provenance blocks.
**Gate:** a checked-in change table (file, field, 0.5.7, 0.7.2) that matches the 10
measured changes in `TRAINING-0.7.2.md`.

**Status 2026-10-17: done (uncommitted).** `tools/diff_assets_072.py` (stdlib, ~13 s)
wrote [`ASSET-DIFF-0.7.2.md`](ASSET-DIFF-0.7.2.md). Assets: 9,909 added, 1,298 removed,
4,286 changed; of 375 simulator-referenced paths present in the installed 0.5,
127 changed and 8 moved. Cross-check: 7/10 measured changes shown field-level, NPC
role moves partial, Trork/Kweebec roles value-identical apart from test-beacon
names. Spot-checked by the main session: `Stats/Mana.json` 0/0 -> 100/100 (regen
interval 0.2 -> 0.3); `MovementConfig/Default.json` JumpForce 11.8 and BaseSpeed 5.5
unchanged, new `MaxSlopeAngleDegrees` 30 and `MaxWallAngleDegrees` 15 (P2 slopes).
**Caveat:** the installed "0.5" is 0.5.10 (jar manifest), and its Assets sha
(6D14386D...) is not the simulator's 0.5.7 pin (1B8802C2...), so this diff is
0.5.10 -> 0.7.2; a 0.5.7 -> 0.5.10 delta is not covered.

### P1 Version plumbing (no behaviour change for 0.5)
- New `B/rulesets/hytale_0_7_2/` (7 files mirroring `hytale_0_5_7/`), loaders keyed
  on the profile instead of hard `"0.5.7"` checks: `rulesets/loader.py:17,43-73`,
  `asset_surface.py:21-28,188-229,346` (`require_hytale_0_5_7_assets_archive`),
  `damage_causes.py`, `role_statuses.py`, `movement_projection.py`,
  `native_status_projection.py`, `combat/assets/catalog.py:34-37`.
- Capacities per profile: `B/jax/crafting/contract.py:15` and
  `packing.py:93-120` (1947, NOT version-keyed today, while
  `worldgen/native_crafting.py` already reads 2047 from `v0_7`) -- fix the
  disagreement.
- Labels from the profile: `census_publication.py` (`HYTALE_VERSION`),
  ~25 `"server_version": "0.5.7"` literals in `worldgen/` and `jax/world/`,
  six `"hytale_version"` spec fields under `jax/combat/**/schema/`,
  `region/contract.py:15,60,173-180` (0.5-only release gate),
  `source_pins.py:15,155-167` (expected version).
- `v0_7.py`: `checkpoint_rulesets` gains `"0.7.2"` once P2/P3 land.
**Gate:** full hytalegym suite on 0.5 identical to a recorded baseline (counts and
failing names), `T/test_server_versions.py` extended for both profiles.

### P2 Locomotion and world (what the navigation skills need)
- Movement constants (`B/jax/combat/types.py:79-158`, stamina 194-199; source
  `Server/Entity/MovementConfig/Default.json`, `Stats/Stamina.json`): re-audit with
  `T/fidelity/surrogate_locomotion_source_audit.py --assets --server-jar` on 0.7.2.
  The flat motion differential already matches (position error 1.1e-5).
- Walk-controller minimum descent rate on slopes and ledges (0.7.2 code change,
  unmeasured): new native differential on a stairs, ledge, and slope fixture, extending
  `adk/tools/native_motion_differential.py` beyond the flat trench.
- Line of sight: 0.7.2 passes non-transparent blocks unless a detail hitbox is
  hit. This changes the physical catch rule (`catch_geometry=physical_body` needs
  sight) and the `hidden` opening profile. Port the rule, add an LOS differential.
- Terrain: block runtime ids renumbered, 46/250 Region sections differ, spider webs
  0.65 to 0.45. Recapture the Region library and traversal library under 0.7.2
  (`examples/capture_native_region.py` runs on 0.7.2, 500/500 sections), rebuild
  path-distance tables (`agent_skills/training_runs/tools/region_path_distance.py`),
  the biome table, task suites, and catch-time horizons.
- Reset timing: first step after reset 34-68 ticks (0.5.10: 30); check native
  episode alignment.
**Gate:** motion differential (flat and slopes) within tolerance (5e-4 position), LOS
differential, the locomotion source audit at a new 0.7.2 baseline, and the reach/pursue
task suites regenerated on 0.7.2 Regions.

### P3 Combat content
Per `TRAINING-0.7.2.md` gameplay changes: hit interrupt, stun, and 0.5 s flinch on 31
weapon attacks (a NEW mechanic; no simulator construct exists); sword knockback
6.0 at (0,1,-1) becomes 4.3 at (0,0.2,-1); dagger damage +50%; ranged damage cut 3-4x
(`_iron_shortbow` 2614, `_iron_crossbow` 2787); durability (`generate_item_durability.py
--write`, after its 0.5.7 pin is profile-keyed); mana 100/100 +1 per 0.3 s
(`mechanics/factory.py:86` default 0); guard elemental/knockback modifiers; the
agent's own horizontal knockback cancelled by the bridge on both versions (model or
document). NPC melee/roles unchanged (verify with the diff, do not re-transcribe).
New `profiles/hytale_0_7_2.py`, `weapon_variants/`, `weapon_archetypes/` modules.
**Gate:** `T/fidelity/test_jax_native_combat_compare.py` and
`test_active_combat_compare.py` re-run against 0.7.2 with new receipts; new
differentials for interrupt/flinch and mana.

### P4 Crafting
Capture the 2047-recipe catalog with `T/fidelity/surrogate_crafting_native_compare.py`
on 0.7.2, new fixture, capacity from the profile (P1).
**Gate:** crafting source audit and native compare on 0.7.2.

### P5 Source-pin certifications
New 0.7.2 baselines for the 11 audits in `B/worldgen/source_pins.py:20`
(perception, locomotion, navigation, doors, mutable_blocks, explosion_mutation,
crafting, actor_world_state, explosion_dynamics, inventory, entity_geometry),
each recorded by its `T/fidelity/surrogate_<name>_source_audit.py` from the 0.7.2
jar and assets. **Gate:** all 11 pass on 0.7.2.

### P6 Contracts, checkpoints, cutover
- New ruleset bytes change `combat_ruleset_sha256` and `hytale_version`, which
  invalidates every checkpoint through `combat_checkpoint_contract_errors`
  (`B/jax/training/checkpoint.py:72,208`) and `combat_contract_stamp`
  (`B/jax/combat/contracts/publication.py:415`). A content change (`CONTENT_SHA256`,
  `identities.content`) moves the stamp digest; checkpoint metadata does not carry it.
- Checkpoint lineage (owner decision 2026-10-17): RETRAIN the flat-to-terrain
  lineage on the 0.7.2 simulator; no re-stamped carry-over.
- Native transfer (`examples/jax_arsenal_native_transfer.py`) exact-current on
  0.7.2 for one skill checkpoint, then a native Region reach scene to watch it.
- Default `HYTALERL_SERVER_VERSION` to 0.7.2; mark 0.5 deprecated (keep its
  files until the owner retires them).
**Gate:** native transfer `execution_valid` with a skill checkpoint on 0.7.2.

## Order and parallelism

P0 then P1 (serial; everything else depends on them). Then P2 and P3 in parallel:
P2 unblocks the navigation skills, and P3 is only needed before combat skills.
P4 and P5 can run alongside P3. P6 comes last. Implementation can go to a Haiku
fleet per folder; each phase's gate is re-run by the main session before it counts.

## Known risks

- No automated Assets-to-simulator pipeline for combat tables; P0's diff bounds the
  manual work.
- The interrupt/flinch mechanic is new, not a parameter change.
- Region recapture changes every terrain identity: path tables, suites, biome
  seeds, and reserved audit Regions must be re-selected (audit worlds stay
  unseen).
- The 0.5 tree has uncommitted contract changes (`observation/v3/policy/layout.py`,
  observation width 8272 vs 8271 in tests), so a clean contract baseline must be
  committed before P1 to know what changed.
