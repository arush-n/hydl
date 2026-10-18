# Gen-7 fidelity check: the JAX simulator against Hytale 0.7.2

Measured 2026-10-17, 21:22-22:30 UTC, by the gen-7 fidelity check. Nothing in the simulator was changed;
this is the gap inventory the recalibration (`SIMULATOR-0.7.2.md`, P1-P6) consumes and the ledger the
skill gates cite (`experimental/neural_agent/skill_agent/FIDELITY.md` rows 1-14). Every number below comes
from a receipt made in this run (`R/` = `/Users/arush/.claude/jobs/31947bd7/tmp/fidelity072/`, outside the
repository) or from a file:line that was read; numbers taken from other documents are labelled as such.

## Ground truth used

| Item | Value | Receipt |
|---|---|---|
| 0.7.2 server | `install/release/.../Server/HytaleServer.jar`, Implementation-Version 0.7.2, revision d769dc3d, SHA-256 `23f71f03...0aff` | `R/00_inputs.txt` |
| 0.7.2 assets | `install/release/.../Assets.zip`, 3,942,760,463 bytes | `R/00_inputs.txt` |
| Control build | the installed "0.5" is **0.5.10** (jar `b63ae21f...`, revision d0f83732), not the 0.5.7 the simulator pins (assets pin `1B8802C2...`, `ASSET-DIFF-0.7.2.md:17`). Every "0.5" comparison below is 0.5.10. | `R/00_inputs.txt` |
| Bridge | `hytale-plugin-0.7.2/build/libs/HytaleRLBridge072-0.1.0.jar`, SHA-256 `686BC4BC...0012` = `server_versions/v0_7.py:14-16` | `R/00_inputs.txt` |
| Repository | HEAD `ca5a5a6` (branch main) at the start, `691e854` at the end (other sessions' commits; none touches a file cited here: every cited simulator file predates the first receipt); `HYTALERL_SERVER_VERSION=0.7.2` for every tool that reads it | `R/00_inputs.txt` |
| Concurrent work | during the run another session committed `hytalegym/sim/README.md` and `sim/gen7/GEN7-PLAN.md` (9e6a34b) and left `sim/gen5/`, `sim/gen7/` (profile, extracted 0.7.2 `data/`) and `tools/content_extract/` uncommitted. Nothing in `jax/`, `envs/`, `worldgen/` or `arena/` imports `hytalegym.sim`, so every "simulator today" cell below is the `jax/` engine; the extracted tables were not audited here | `grep` over those trees |
| Decompiled sources | `/Users/arush/hydl/decompiled` is a **pre-0.7.2** decompile (`World` still implements `ChunkAccessor`, `ChunkFlag.TICKING` present, no `Ability4`); 0.7.2 code was read from the jar with `javap` where needed | `R/00_inputs.txt`, `R/javap/` |
| Live server | one 0.7.2 server started with `tools/launch_bridge_072.py --bridge-port 5556 --game-port 25572`; booted in 5.6 s, bridge listening; stopped at the end | `R/live/launcher.log`, `R/live/run1/server.log` |
| Machine | 18 cores, load average 6-10 from other sessions during the run; some world ticks ran long (up to 0.0352 s instead of 1/30 s), which matters for row 1c | `R/live/motion/*.json` (`actions[].dt_seconds`) |

## Summary

**Status over the 36 rows of table (a): MATCH 5, MISMATCH 14, ABSENT 9, UNMEASURED 8.** The live 0.7.2
server started (5.6 s boot) and ran for 46 min: smoke 11/11 checks; motion differential 8 runs; the
geometry and line-of-sight gate (pass); crafting catalog capture; reset timing and NPC activation
samples; 54 pytest combat compares (12 pass).

1. **No source pin holds on 0.7.2.** 0/11 audits pass on 0.7.2 and 11/11 pass on the 0.5.10 control, so
   every failure is 0.7.2 drift. Four audits do not even produce a report: they crash on classes 0.7.2
   removed (`LocalCachedChunkAccessor`, `BlockAccessor`, `BlockHealthModule$BlockHealthSystem`) or on NPC
   role assets that moved to `Faction/` and `Instructions/Combat/`. Re-baselining them (P5) needs code
   changes, not only new hashes.
2. **Locomotion values did not change; the trained model has no oracle.** All 21 player locomotion
   constants in `types.py` equal the 0.7.2 `MovementConfig/Default.json`; 0.7.2 only adds
   `MaxSlopeAngleDegrees` 30 and `MaxWallAngleDegrees` 15, which the simulator lacks. The live flat walk
   plus a 3-block ledge drop matches to 1.1444e-5 m (same number as on 0.5.9). But the skills train with
   `motion_model="player"` (5.5 m/s, jump 11.8), while the bridge body is an NPC on `MotionControllerWalk`
   (5.0 m/s on the live receipt): the player model cannot be certified through the bridge at all.
3. **NPC jumps fail on 0.7.2** (0.0044 m in place, 0.30 m and 0.70 m with forward motion). The divergence
   starts on the first world tick longer than 1/30 s: native gravity uses the previous tick's duration,
   JAX the current one (inferred from the per-step velocity deltas). On 0.5.9 the same runs failed at
   landing instead.
4. **Line of sight agrees on full blocks only.** 3/3 fixture rays (wall, diagonal graze, diagonal block)
   agree on 0.7.2; the 0.7.2 rule (opaque blocks occlude only where a detail hitbox is hit; confirmed in the
   0.7.2 `PositionCache` constant pool) is not ported and no bridge fixture tests it.
5. **Melee:** the hit interrupt/stun/flinch construct is absent, but by the 0.7.2 role graph it does not
   apply to the simulator's opponents (Trork_Brawler, Trork_Unarmed, Kweebec_Razorleaf carry no
   eligibility; skeletons and goblins do). Three sword swings (knockback), daggers (+50 % damage), guard's
   attack-cancel rule and the durability table (57 maximum and 82 loss-per-hit rows of 143) are stale; the
   agent's own hits land 1-2 ticks later than pinned. Trork activation (first turn 44, first
   root 46) and hit timing ([16,16,15,16,15] ticks) on 0.7.2 equal the 0.5.7 pins.
6. **Ranged:** shortbow tiers 0-3 launch speed and damage are stale (12/22/35/50 vs 15/30/50/70; 1/2/4/7 vs
   6/9/12/16); full draw 85/25, crossbow 40/10 and Big Arrow 100/40 launch values are unchanged.
7. **Magic:** mana is 0 by default (0.7.2: 100, +1 per 0.3 s) and no rune exists in the simulator.
8. **Crafting:** the live 0.7.2 catalog has 2,047 recipes; the JAX table holds 1,947 and its packer
   rejects the 0.7.2 catalog (988 excluded-material inputs).
9. **Reset timing:** the "34-68 ticks" figure did not reproduce; the first step after reset executed 4
   engine ticks (1 in the motion harness), with one extra world tick in `kill_trork` sessions.
10. **Combat compares on 0.7.2: 12 of 54 pass** (`tests/fidelity/combat/` 7/42, selected 5/12). The failures
   that measure 0.7.2 content: daggers 5.0 vs 3.0 damage; shortbow tier 0 damage 1.0 vs 6.0; the agent's
   hits land 1-2 ticks later (iron sword tick 9 vs 7, kunai one tick per throw); in 12 duel runs the native
   arrow or bolt registered no hit where JAX predicts one (cause not isolated). The rest are harness or
   setup: 14 learner-row raw-group differences (cause not isolated), 6 stale head names in the crossbow
   helpers, 3 actor-to-actor runs where no attack landed (one native body yaw stuck at -180°), 1 bridge
   error, 2 refusals on the 0.5.7 assets pin. Dodge (2 tests), shield bash, spear throw, three seeded
   melee contacts and the Trork timing agree.

## (a) Mechanics of BUILD.md section 4, measured

Rows follow BUILD.md section 4 (= `skill_agent/FIDELITY.md` seed rows 1-14), split where the evidence
differs. "§" is the `SKILL_CATALOGUE.md` section. Simulator paths are under
`HytaleRL/hytalegym/hytalegym/` unless they start with `HytaleRL/`, `adk/` or `arena/`; test paths under
`HytaleRL/hytalegym/tests/`. Status: MATCH (live or static agreement shown), MISMATCH (simulator differs
from 0.7.2), ABSENT (no construct), UNMEASURED (no live number; a static agreement is noted).

| # | Mechanic (catalogue) | Expected on 0.7.2 | Simulator today | Live 0.7.2 measurement | Status | Pass bar | Phase |
|---|---|---|---|---|---|---|---|
| 1a | Player gaits and jump (§1 walk_to, run_to, sprint_to, sneak_to, step_and_hop) | walk 1.65, run 5.5, sprint 7.0, sneak 3.0, jump 11.8: BaseSpeed 5.5 × 0.3 / 1.0 / 1.273 / 0.55 = 1.65 / 5.5 / 7.0015 / 3.025, JumpForce 11.8 (`MovementConfig/Default.json`, unchanged from 0.5.10) | `jax/combat/types.py:79-160`: all 21 player constants equal the 0.7.2 file (`R/02_locomotion_constants_vs_assets.md`). Training uses this model: `default_combat_params(motion_model="player")` (`types.py:549`) called by `arena/jax_env.py:199`, `arena/training/runs/pursuit_run.py:1334`, `jax/training/multi_actor/cli.py:240` | not measurable: the bridge body is an NPC role on `MotionControllerWalk` (`types.py:180-185`); player transition physics is client-side (`tests/fidelity/surrogate_locomotion_source_audit.py:234-245`: `transition_physics_location: not_present_in_server_jar`). The live body ran at 5.0 m/s, not 5.5 (`R/live/motion/forward-30.json`, Kweebec_Razorleaf) | UNMEASURED (static values match) | position error <= 5e-4 per step | P2 |
| 1b | NPC walk, flat floor plus a 3-block ledge drop (§1; the native-certified model) | native = JAX | `motion_model="npc_walk"` (`types.py:587-598`, `jax/combat/runtime/agent.py:604`) | `adk/tools/native_motion_differential.py --steps 30`: max position error **1.1444e-5 m**, velocity 6.2e-4, pass (`R/live/motion/forward-30.json`; 0.5.9 receipt `adk/runs/native-motion-differential-main-20260905.json`: 1.1444e-5). `native_geometry_compare.py` ledge walk: 1.85e-5 m, airborne tick 12, landing 23, pass (`R/live/03c_geometry.log`). `--steps 35` fails only by JAX geometry-window exhaustion at state 35 (`R/live/motion/forward-35.json`), as 0.5.9 did at state 36 | MATCH | <= 5e-4 | P2 |
| 1c | NPC jump (§1 step_and_hop) | native = JAX | npc_walk jump velocity sqrt(2 g h) = 5.099 (`types.py:592-598`) | all fail (`R/live/motion/`): jump in place 12 ticks 4.40e-3 m / 0.132 m/s; single jump + forward 30 ticks **0.703 m**; jump + forward 30 ticks **0.304 m**. First failing state = first tick longer than 1/30 s (states 4, 2, 2; those runs' ticks reached 0.0350 s). Per-step deltas: on a long tick native Δv = -32 × (1/30) and on the next tick -32 × 0.0346, i.e. native gravity uses the previous tick's dt; JAX uses the current one (inference from the receipts). On 0.5.9 (ticks nominal to within 1e-5 s) the same variants failed at landing, state 11: 0.304 / 0.359 m (`storage/runs/shared/basic/20260927T011329Z-8c08f2e2/artifacts/native-motion-*`) | MISMATCH | <= 5e-4 | P2 |
| 1d | Slope and wall limits (§1 go_around) | `MaxSlopeAngleDegrees` 30, `MaxWallAngleDegrees` 15 (new in 0.7.2; `ASSET-DIFF-0.7.2.md:100-106`) | absent (no constant; `R/02_locomotion_constants_vs_assets.md`) | not measured: the bridge has no slope or stairs fixture (`hytale-plugin-0.7.2/.../NativeEnvironmentSession.java:8615-8640`) | ABSENT | <= 5e-4 on a slope fixture | P2 |
| 1e | Walk controller's minimum descent rate (0.7.2 code; `TRAINING-0.7.2.md` item 8) | native = JAX when walking down | 0.5 NPC walk | a 3-block drop off a ledge matches (1b); walking down slopes or stairs not measured (no fixture) | UNMEASURED | <= 5e-4 | P2 |
| 1f | Slide, landing roll, automatic step-up (§1 slide, land_safely, step_and_hop) | slide 6.99 → 2.5; roll at fall speed 21-31, full protection <= 25; step-up obstacles <= 45° (all unchanged from 0.5.10) | absent for the player model: 44 `MovementConfig` keys have no simulator constant (`R/02_locomotion_constants_vs_assets.md`); the NPC model has only step-up/drop heights (`types.py:274-275`, `runtime/agent.py:1034-1042`) | not measurable (player client features, see 1a) | ABSENT | present and matching | P2 |
| 2a | Line of sight, full-cube occluders (§2 search_for, track; §8 hide) | native = JAX on every ray | opaque unit cells, `jax/world/geometry/__init__.py:260-339` | `tests/fidelity/native_geometry_compare.py`: LOS wall (blocked), diagonal graze (visible), diagonal block (blocked) all agree, 3/3 (`R/live/03c_geometry.log`) | MATCH | every fixture ray | P2 |
| 2b | Line of sight through opaque blocks with partial hitboxes (slabs, stairs, fences) | occluded only where a detail hitbox is hit (`PORT-0.7.2.md`, "Server rules that changed"); the 0.7.2 `PositionCache` newly references `BlockBoundingBoxes$RotatedVariantBoxes.getDetailBoxes`, `BlockType.getHitboxTypeIndex`, `CollisionMath.intersectVectorAABB` (absent from 0.5.10; `R/javap/PositionCache_{0510,072}.txt`) | occludes the whole opaque cell (`geometry/__init__.py:266-284`), the 0.5 rule | not measured: no bridge fixture puts a partial-hitbox opaque block in a ray | MISMATCH (static) | every fixture ray | P2 |
| 3a | Terrain: Region library, block ids, sections | library recaptured on 0.7.2 (`TRAINING-0.7.2.md:47` reports 500/500 sections captured on 0.7.2; `:138-140` 46/250 sections differ, Rock_Stone id 1351 → 1530) | every staged library was captured on server 0.5.7: `HytaleRL/artifacts/worldgen/region-library-pilot-v2/manifest.json` (`capture_contract.server_version` "0.5.7"), `region-library-v3-288`, `region-library-pilot-v1` | not recaptured in this run | MISMATCH | 500/500 sections; suites, path tables, horizons rebuilt | P2 |
| 3b | Block movement settings (§8 avoid_hazards) | spider web horizontal speed 0.65 → 0.45, jump 0.5 → 0.45, drag 0.45 → 0.82 (`ASSET-DIFF-0.7.2.md` cross-check row 7) | taken from the 0.5.7 captures (no hard-coded value) | not measured | MISMATCH (static) | through 3a | P2 |
| 4 | Reset timing and NPC activation | "first step after reset 34-68 ticks" (`TRAINING-0.7.2.md:141`) | target activation tick drawn from [36, 44] (`rulesets/hytale_0_5_7/kweebec_razorleaf_vs_trork_brawler_v9.json:122-123`, `jax/combat/runtime/reset.py:128-132`); JAX contract pins first turn 44, first root 46 (`tests/fidelity/test_jax_native_combat_compare.py:4241-4242`) | first step after reset executes 4 engine ticks (survive, kill_trork; `R/live/03f_reset_timing.json`, `R/live/03f_reset_timing_kill_trork.json`) or 1 (motion harness, reset at world tick 20); in kill_trork the world tick advances 5 on the first step (one tick outside the executed count), 4 after. Trork first turn **44**, first root **46** in 5/5 samples over 2 seeds (`R/live/03g_root_ticks.json`). The 34-68 figure did not reproduce | MATCH (activation) | documented offset applied in the session clock | P2 |
| 5a | Hit interrupt, stun and flinch (§12 interrupt; BUILD's "new construct") | 31 interaction files parent `DamageEntityParent_Hit_Interrupt` (0.5.10: none); Flinch 0.5 s at horizontal speed × 0, Stunned 0.7 s, attack cancelled, Enraged 8 s after the third flinch (`R/02_static_mechanics_receipts.txt`) | absent: no `Hit_Interrupt` or flinch anywhere in `hytalegym/`; the arsenal's `InterruptedBy` mask covers only an actor's own inputs (`jax/combat/arsenal/schema/types.py:134-137`) | not measured live. Static: eligibility is applied only by `Component_Instruction_Hit_Interrupt` (22 role files, e.g. Template_Intelligent, Template_Predator, goblins, skeletons, spiders); the reference chains of Trork_Brawler, Trork_Unarmed and Kweebec_Razorleaf never reach it, those of Skeleton_Fighter and Goblin_Scrapper do (`R/02_static_mechanics_receipts.txt`). Inferred: hits do not interrupt the simulator's current opponents | ABSENT | flinch timing and displacement match | P3 |
| 5b | Sword knockback (§9 sword) | Swing_Left 4.3 at (0, 0.2, -1); Swing_Right 5.0 at (0, 0.2, -1); Swing_Down 10.8 at (0, 0.2, -2); thrust and Vortexstrike unchanged (`R/02_weapon_asset_delta.txt`) | 0.5 values: `jax/combat/arsenal/profiles/hytale_0_5_7.py:2064` (0, 1, -1, 6.0), `:2094` (0, 1, -1, 7), `:2108` (0, 1, -2, 12) | not measured in this run (`TRAINING-0.7.2.md:93-99` reports Trork displacement 0.19 m on 0.7.2 vs 0.44 m on 0.5.10) | MISMATCH | per-hit displacement within tolerance | P3 |
| 5c | Dagger damage (§9 daggers) | Iron swings 4 → 6, Mithril 8 → 12 (`R/02_weapon_asset_delta.txt`) | `hytale_0_5_7.py:2521` damage (4, 4, 8, 12, ...) | seeded bronze-dagger contact: native damage **5.0** vs JAX 3.0 (`R/live/03h_pytest_combat_summary.txt`) | MISMATCH | damage exact | P3 |
| 5d | Guard cancelling an attack; shield bash (§9 shield_guard, §12 guard, bash) | 0.7.2 removes `Rules.Interrupting: [Primary]` from the Sword, Battleaxe, Daggers and Mace guard roots, gives the sword thrust `InterruptedBy: [Secondary]`, and changes shield bash `StaminaRegenDelay` from Set -1.5 to Min -1 (`ASSET-DIFF-0.7.2.md:237-243, 266-322, 1072-1127`) | 0.5.7 rule: a guard root interrupts Primary (`jax/combat/arsenal/programs/interaction_rules.py:117-141`, mask from `hytale_0_5_7.py:1010`) | `test_native_shield_bash_resource_cost_matches_and_regen_phase_is_explicit` passes (bash cost and regen phase agree with JAX; `R/live/03i_pytest_selected_summary.txt`); the guard attack-cancel rule was not exercised by any test run | MISMATCH (static: guard rule) | lifecycle ticks match | P3 |
| 5e | Melee timing and damage where 0.7.2 did not change them (§9) | Trork battleaxe hits at elapsed ticks [16, 16, 15, 16, 15] (0.5.7 pin, `tests/fidelity/test_active_combat_compare.py:40`) | same pin in the JAX contract | 0.7.2: [16, 16, 15, 16, 15] in 5/5 samples (`R/live/03g_root_ticks.json`); `test_pure_jax_combat_fidelity_contract` passes (JAX first turn 44, first root 46). Seeded melee contacts of a cobalt battleaxe, a scrap NPC mace and a thorium sword agree with JAX in damage and hit tick (3 passes, `R/live/03h_pytest_combat_summary.txt`) | MATCH | exact ticks and damage | P3 |
| 5f | The agent's own attack and throw dispatch timing (§9 sword, daggers; §10 throw_kunai) | hit ticks as pinned on 0.5.7: iron sword left swing at tick 7 (`tests/fidelity/test_jax_native_combat_compare.py:4318`); kunai roots every 10 ticks | JAX: iron sword tick 7 (both seeds), kunai [0, 10, 20, 30, 60, 90, ...] | 0.7.2 native: iron sword damage equal but hit at tick **9** (`R/live/03i_pytest_selected_summary.txt`); bronze daggers tick 18 vs JAX 17; kunai roots [0, **11, 22, 33, 62, 91**, ...] (`R/live/03h_pytest_combat_summary.txt`). Long ticks would make hits land earlier in tick count, not later, so load does not explain it (inference) | MISMATCH | exact ticks | P3 |
| 6a | Shortbow draw tiers 0-3 (§10 shortbow) | launch 12 / 22 / 35 / 50, gravity 45 / 40 / 32 / 28; Iron damage 1 / 2 / 4 / 7 (`R/02_projectile_config_delta.txt`, `R/02_weapon_asset_delta.txt`) | launch 15 / 30 / 50 / 70, gravity 45 / 40 / 32 / 28, damage 6 / 9 / 12 / 16 (`hytale_0_5_7.py:2617, 2623, 2629-2630`) | `test_native_shortbow_projectile_contact_matches_jax_open_and_wall`: native damage **1.0** vs JAX 6.0 at tier 0 (`R/live/03i_pytest_selected_summary.txt`). `test_native_shortbow_outer_root_strengths_match_jax` (tiers 0-4) and five seeded shortbow variants: the native arrow registered **no hit** in all 10 runs where JAX predicts hits at ticks 14-44 | MISMATCH | range error <= 2 % per tier | P3 |
| 6b | Bow full draw, crossbow bolt, Big Arrow launch (§10 shortbow, crossbow) | 85 / 25, 40 / 10, 100 / 40, all unchanged (`R/02_projectile_config_delta.txt`) | 85 / 25 (`hytale_0_5_7.py:2629-2630`), 40 / 10 (`:2803-2804`), 100 / 40 (`:2866-2867`) | iron spear throw contact (open and wall) agrees with JAX (pass); Big Arrow contact obeys its damage bound (2 passes, but a miss also passes); the iron crossbow combo's three accepted shots and the full-draw shortbow tier registered **no hit** natively (cause not isolated: aim, contact or target behaviour) | UNMEASURED (static launch values match; live contact inconclusive) | range error <= 2 % per tier | P3 |
| 6c | Crossbow damage (§10 crossbow) | Iron combo 27 → 13; Ancient Steel combo 42 → 14, Big Arrow 123 → 82, standard 16 → 10.5 (`R/02_weapon_asset_delta.txt`) | `hytale_0_5_7.py:2790` (10, 27, 78) | not measured: the iron crossbow shots did not hit; the Ancient Steel tests stop on a stale head name (see the pytest table) | MISMATCH (static) | damage exact | P3 |
| 7a | Mana (§11) | 100 / 100, +1 per 0.3 s while alive (0.5.10: 0 / 0, 0.2 s with a 6 s no-damage delay) (`R/02_static_mechanics_receipts.txt`) | `jax/combat/mechanics/factory.py:86` maximum 0.0 by default; `:133-134` +1 per 0.2 s | not measured in this run (`TRAINING-0.7.2.md:113` reports 100/100) | MISMATCH | rates within 5 % | P3 |
| 7b | Runes (§11 fireball, wind_strike, chain_hook, ground_slam, charged_frost_shot, enrage, poison_imbue) | 7 ability runes plus `Rune_Empty` and 11 modifier runes under `Server/Item/Items/Rune/` | absent (no rune in `hytalegym/`) | not measured | ABSENT | cost, cooldown, cast time, flight, effect | P3 |
| 8a | Stamina (§8 recover_stamina; sprint, guard, dodge) | 10, minimum -4, +0.3 per 0.1 s, sprint -0.1 per 0.1 s; 0.7.2 adds `UseIgnoreFlag` to the "not wielding" regeneration condition (engine meaning not read) | `types.py:194-200`, `jax/combat/mechanics/schema/contract.py:123-129`: equal | regeneration and sprint drain not measured; shield-bash stamina cost agrees (5d's pass) | UNMEASURED (static values match) | rates within 5 % | P3 |
| 8b | Oxygen (§1 dive_and_breathe) | 100, +25 per 0.5 s breathing, -3 per 0.5 s suffocating (identical to 0.5.10) | `mechanics/schema/contract.py:135-139`: equal | not measured (no breathing differential) | UNMEASURED (static values match) | rates within 5 % | P3 |
| 8c | Signature energy (§9 signatures) | Vortexstrike costs 100 % of the signature bar (`ValueType: Percent`), unchanged | bar and cost 20 = 100 % (`hytale_0_5_7.py:2184-2197`) | not measured | UNMEASURED (static values match) | rates within 5 % | P3 |
| 8d | Durability tables (§6 repair_gear; every weapon) | `Weapon_Sword_Iron` 250, 0.13 per hit | `jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json:796-801`: 120, 0.21, generated only from the pinned 0.5.7 zip (`HytaleRL/scripts/combat/assets/generate_item_durability.py:1,31`). Of its 143 weapon rows, **57** differ in maximum and **82** in loss per hit on 0.7.2 (`R/02_durability_delta.txt`) | not measured live | MISMATCH | table equality | P3 |
| 9 | Dodge (§1 dodge) | left and right only, force 13, 0.25 s invulnerable, 2 stamina, regen delay -0.7; all 7 dodge assets identical to 0.5.10 (so "left/right only" is not new in 0.7.2) (`R/02_static_mechanics_receipts.txt`) | `jax/combat/mechanics/schema/contract.py:70-81` (forward/back masked by `DODGE_AUTHORED_ACTION_MASK = (False, False, True, True)`), `:164-178` (13.0, 0.25, 2.0, -0.7); the policy head still has 4 dodge slots (`jax/combat/observation/v3/schema/contract.py:123`, `observation/v3/policy/layout.py:194-199`) | `test_native_policy_dodge_boundary_matches_advertised_capability` and `test_native_lateral_dodge_projects_mechanics_when_supported` pass (`R/live/03i_pytest_selected_summary.txt`); `test_live_lateral_dodge_blocks_overlapping_native_hit` did not measure the dodge (its control attack never landed) | MATCH | impulse and timing match | P3 |
| 10a | Climbing (§1 climb) | 44 climbable blocks; climb 0.035, sprinting up 0.07 | absent: the `climbing` movement-state bit is always written False (`jax/world/entities/movement_states.py:344`); traversal "climb" edges are NPC step-ups (`worldgen/region/traversal.py:45`) | not measured | ABSENT | present and matching | new (after P2) |
| 10b | Doors (§5 place_furniture; navigation) | door open/close transitions | surrogate present (`worldgen/surrogate/stateful.py`, `jax/world/surrogate/features/interactions.py`); no door head on the 12-head v3 policy surface (`observation/v3/policy/layout.py:201-259`) | door source audit fails on 0.7.2 (row 12); `tests/fidelity/surrogate_door_native_compare.py` refuses unless that audit matches 0.5.7 (`:104-105`); run as shipped on 0.7.2 it stops earlier, inside its embedded source audit: `KeyError` for `BlockAccessor.class` (`R/live/03j_door_compare.out`) | UNMEASURED | present and matching | P5, then P2 |
| 10c | Swimming and diving (§1 swim_to, dive_and_breathe) | player swim × 0.6, up and down 2.5, sinking 1.35 | NPC dive only (`jax/world/entities/swimming.py:112`); no player swimming | not run: `tests/fidelity/surrogate_swimming_native_compare.py:79` asserts `native_server_version == "0.5.7"` | ABSENT (player swimming) | present and matching | new (after P2) |
| 10d | Gliding (§1 glide) | glider speed 10, falls <= 2/s, stamina 1/s | absent (only a `gliding` state flag, `jax/world/entities/physical_state.py:91-129`) | not measured | ABSENT | present and matching | new |
| 10e | Riding (§1 ride) | rider speed 8, sprint 13.2, jump 14 (catalogue) | absent | not measured | ABSENT | present and matching | new |
| 11 | Crafting (§7; kind C) | 2,047 recipes, 19 benches, cook 2 s | table capacity 1,947 (`jax/crafting/contract.py:15`, `jax/crafting/packing.py:116`); the native catalog check is profile-keyed (`worldgen/native_crafting.py:18, 205`) | live capture (`R/scripts/crafting_catalog_capture.py`): **2,047** recipes, same semantic hash on seeds 570057 and 1194077737; **22** distinct bench ids (incl. `TODO` with 58 recipes, `Armor_Bench`/`ArmorBench`, `Architects`/`Architectsbench`); campfire 3 recipes at **2.0 s**; `current_jax_packer_ready` False (988 excluded-material inputs, 19 recipes without a bench) (`R/live/03e_crafting_capture.json`). The shipped `surrogate_crafting_native_compare.py` refuses: "installed crafting sources differ from Hytale 0.5.7" (`R/live/03d_crafting_compare.out`) | MISMATCH | catalogue and timings match | P4 |
| 12 | The 11 source-pin audits | 11/11 pass | pinned to 0.5.7 (`worldgen/source_pins.py:20-76`) | **0/11** on 0.7.2, 11/11 on 0.5.10 (table (b)) | MISMATCH | 11/11 on 0.7.2 | P5 |
| 13 | Contracts and checkpoints | new ruleset bytes change the stamp; one skill transfers on 0.7.2 | stamp at `jax/combat/contracts/publication.py:406`; only `rulesets/hytale_0_5_7/` exists; `server_versions/v0_7.py:18` accepts only 0.5.7 checkpoints | no gen-7 checkpoint exists to transfer | ABSENT | native transfer `execution_valid` | P6 |
| 14 | Legal observation | no privileged field reaches a body | `jax/combat/observation/v3/encoding/encoder.py:550-565`, off by default (`jax/combat/arsenal/runtime.py:198`, `arsenal/schema/types.py:513`); its perceptibility uses row 2's line of sight | no native differential is defined for it | UNMEASURED | three-world leakage test | skill_agent |

### Live combat compares on 0.7.2 (`tests/fidelity/combat/`, all 42 native tests)

`pytest -m native_server hytalegym/tests/fidelity/combat` on the live 0.7.2 bridge (lease redirected,
`--timeout 600`): **7 passed, 35 failed** in 28 min (`R/live/03h_pytest_combat.{log,xml}`, summary
`R/live/03h_pytest_combat_summary.txt`). By cause, from each test's failure text:

| Tests | Result | What it measured on 0.7.2 |
|---|---|---|
| crossbow reloads six arrows at 30 tps; queue drain keeps every learner head satisfiable | 2 pass | reload lifecycle and action-queue mask agree with JAX |
| melee contact, seeded variants: cobalt battleaxe, scrap NPC mace, thorium sword | 3 pass | damage and hit tick agree with JAX (row 5e) |
| Big Arrow contact obeys the damage bound (template 46, iron 78) | 2 pass | upper bound only: a miss also passes, so weak evidence for 6b |
| melee contact, bronze daggers | 1 fail | native damage **5.0** vs JAX 3.0; hit tick 18 vs 17 (row 5c) |
| ranged contact, seeded shortbow variants, charge tiers 0-4 | 5 fail | native recorded **no hit** (damage 0) in all five; JAX predicts hits of 5, 14, 12, 13, 15 at ticks 14-44 (row 6a; cause not isolated: tiers 0-3 launch slower on 0.7.2, tier 4 does not) |
| iron crossbow three-contact combo (2 seeds) | 2 fail | reload and three shots accepted, **no hit** registered, run truncated (rows 6b, 6c) |
| kunai charge bank, root-relative | 1 fail | native roots at ticks [0, 11, 22, 33, 62, 91, ...] vs pinned [0, 10, 20, 30, 60, 90, ...]: one tick later per throw (§10 throw_kunai) |
| lateral dodge blocks an overlapping hit; group combat actor independence; nonzero actor executes ability, guard, dodge | 3 fail | the control attack never landed (dodge test stops at `next(...)` on a health drop, `test_native_dodge_hit_overlap.py:269`); no damage in any group arm; a native body yaw stayed at -180° against a commanded 0°. The dodge itself was not measured (row 9) |
| dynamic learner row vs source-aligned JAX (14 cases) | 14 fail | action mask equal (124 wide, 0 mismatches); raw row differs in a `world_geometry` group (52 values, one side 0.0, the other 0.5 / 0.333 / 1.0 / 0.004 / -0.25) and at indices 6200+; cause not isolated (candidates: the uncommitted observation-contract changes the plan names, 0.7.2 world content) |
| Ancient Steel crossbow combo (2); crossbow signature lifecycle (4) | 6 fail | harness break, not 0.7.2: the targeting helper asks for head `world_move_none_plus_compass` (`tests/fidelity/combat/projectiles/crossbow/test_native_crossbow_combo.py:255`, `.../test_native_crossbow_signature_lifecycle.py:89`), which the current 12-head layout no longer has |
| crossbow primary interrupts reload | 1 fail | bridge error: `IllegalThreadStateException: World thread is not accepting tasks` |
| flame staff static placement (2 seeds) | 2 fail | this run's setup error (`HYTALE_ASSETS_ZIP` not exported); rerun with it set (`R/live/03l_flame_staff.log`), both refuse at the asset gate: "Hytale 0.5.7 Assets.zip SHA-256 mismatch ... expected=1B8802C2..., actual=EE8F8D5B..." (the 0.7.2 zip) |

What each needs to run on 0.7.2 as a gate: the targeting helpers updated to the current head names; a
0.7.2 calibration of the expected values (daggers, kunai timing, shortbow tiers, crossbow damage); an
explanation of the missing projectile and actor-to-actor hits (facing, aim or contact) before any ranged
or group differential can certify; and, for `test_active_combat_compare.py`'s full differential, a
`SimulatorBridgeMain` listener on port 5557 (only its native-only `--root-tick-samples` mode was run).

**Selected compares from the other fidelity files** (`R/live/03i_pytest_selected.{log,xml}`,
`R/live/03i_pytest_selected_summary.txt`): **5 passed, 7 failed** in 4.5 min.

| Test | Result | Measurement |
|---|---|---|
| `test_jax_native_combat_compare.py::test_pure_jax_combat_fidelity_contract` (no server) | pass | JAX Trork first turn 44, first root 46 (row 4) |
| `::test_native_iron_sword_program_matches_jax_damage_and_hit_tick` | fail | damage equal; native hit tick **9** vs pinned 7 (row 5f) |
| `::test_native_policy_dodge_boundary_matches_advertised_capability` | pass | row 9 |
| `::test_native_lateral_dodge_projects_mechanics_when_supported` | pass | row 9 |
| `::test_native_shortbow_outer_root_strengths_match_jax` (tiers 0-4) | 5 fail | native no hit; JAX damage 6 / 9 / 12 / 16 / 19 at ticks 14 / 18 / 27 / 35 / 44 (rows 6a, 6b) |
| `::test_native_shield_bash_resource_cost_matches_and_regen_phase_is_explicit` | pass | rows 5d, 8a |
| `test_arsenal_projectile_native_compare.py::test_native_shortbow_projectile_contact_matches_jax_open_and_wall` | fail | native damage **1.0** vs JAX 6.0 at tier 0 (row 6a) |
| `test_arsenal_projectile_native_compare.py::test_native_iron_spear_throw_contact_matches_jax_open_and_wall` | pass | spear throw contact (§10 throw_spear) |

Also run on the live server: three more jump-in-place repeats, all failing (3.8e-3, 4.7e-3, 8.4e-3 m;
first failure at state 2, each run with long ticks in its first three steps; `R/live/motion/jump-in-place-12-repeat{1,2,3}.json`).

## (b) The eleven source-pin audits on 0.7.2

Each audit ran exactly as `worldgen/source_pins.py:106-122` runs it (same flags, cwd and `PYTHONPATH`;
`HYTALE_ASSETS_ZIP` set because the scripts compute their defaults at parse time), driven by
`R/scripts/run_source_audits.py`. Besides the audit's own output, the driver looked up and hashed every
pinned class and asset, so a crash still yields the full drift list. Raw output: `R/audits/072/*.stdout.json`,
`*.stderr.txt`, `*.analysis.json`; control: `R/audits/0510/`; log `R/01_source_audits.log`.

| # | Audit | 0.7.2 exit / result | Pinned classes byte-equal on 0.7.2 | Pinned assets equal on 0.7.2 | First mismatch on 0.7.2 | 0.5.10 control |
|---|---|---|---|---|---|---|
| 1 | perception | 1 / FAIL (crash) | 2/26, 1 removed | not reached | `KeyError`: `.../accessor/LocalCachedChunkAccessor.class` removed | 0 / PASS (26/26) |
| 2 | locomotion | 1 / FAIL | 8/22 | 1/7 (MovementConfig and GameplayConfigs Default, Template_Swimming_Passive, 3 test roles changed) | `class_sha256`: 14 changed, first `protocol/MovementStates.class`; dive role assets 5 → 6 (`Test_Dive_Stationary.json` added) | 0 / PASS (22/22) |
| 3 | navigation | 1 / FAIL (crash) | 6/18 | 0/4 (all 4 moved) | `KeyError`: `Server/NPC/Roles/Intelligent/Aggressive/Trork/Trork_Brawler.json` (moved to `Faction/`) | 0 / PASS (18/18) |
| 4 | doors | 1 / FAIL (crash) | 1/6, 1 removed | 9/9 | `KeyError`: `.../accessor/BlockAccessor.class` removed | 0 / PASS (6/6) |
| 5 | mutable_blocks | 1 / FAIL (crash) | 19/78, 1 removed | not reached | `KeyError`: `blockhealth/BlockHealthModule$BlockHealthSystem.class` removed | 0 / PASS (78/78) |
| 6 | explosion_mutation | 1 / FAIL | 0/12 | 1/1 | `class_sha256`: 12 changed, first `math/block/BlockSphereUtil.class` | 0 / PASS (12/12) |
| 7 | crafting | 1 / FAIL | 8/24 | recipe corpus changed | `class_sha256`: 16 changed, first `crafting/component/BenchBlock.class`; recipe assets 403 + 1,544 = 1,947 → 410 + 1,637 = **2,047**; knowledge-required 19 → 25; resource-selector inputs 974 → 988 | 0 / PASS (24/24) |
| 8 | actor_world_state | 1 / FAIL | 4/17 | (jar only) | `class_sha256`: 13 changed, first `asset/type/fluid/Fluid.class` | 0 / PASS (17/17) |
| 9 | explosion_dynamics | 1 / FAIL | 0/2 | (jar only) | `class_sha256`: 2 changed, first `entity/ExplosionUtils.class` | 0 / PASS (2/2) |
| 10 | inventory | 1 / FAIL | 0/4 | (jar only) | `class_sha256`: 4 changed, first `inventory/Inventory.class` | 0 / PASS (4/4) |
| 11 | entity_geometry | 1 / FAIL | 3/5 | (jar only) | `class_sha256`: 2 changed, first `entity/component/BoundingBox.class` | 0 / PASS (5/5) |

Every 0.5.10 run passed because the 0.5.7-pinned classes and assets are byte-identical in 0.5.10
(`R/audits/0510/*.analysis.json`, e.g. 78/78 classes for mutable blocks). On 0.7.2 the class drift is near total,
so the P5 baselines are new recordings, and audits 1, 3, 4 and 5 first need their pinned class and asset
paths updated.

**Locomotion audit against `types.py` and the catalogue** (`R/02_locomotion_constants_vs_assets.md`):
the audit fails on class bytes and on the two `Default.json` files, but the only value-level change in
`MovementConfig/Default.json` is the two added keys (`MaxSlopeAngleDegrees` 30, `MaxWallAngleDegrees`
15); all 21 values the simulator transcribes (base speed, the forward/backward/strafe tables, sprint,
jump, variable-jump fall force, air drag and air control) are unchanged and equal `types.py:79-160`. The
catalogue's walk 1.65 / run 5.5 / sprint 7.0 / sneak 3.0 / jump 11.8 are these values (sprint 7.0015,
sneak 3.025 before rounding). Stamina 10 matches `Stamina.json` (max 10, min -4, +0.3 per 0.1 s, sprint
-0.1 per 0.1 s). 44 `MovementConfig` keys have no simulator constant (climb, slide, roll, auto-jump,
fly, swim jump, air friction, jump buffer, slopes and walls).

**Crafting source audit offline:** runs offline (assets plus jar) and fails as above; the live catalog is
in row 11.

## (c) Changed asset files the simulator references, by mechanic

From `ASSET-DIFF-0.7.2.md` (0.5.10 → 0.7.2; 127 changed and 8 moved simulator-referenced files), grouped
by `R/scripts/asset_diff_groups.py` (`R/02_asset_diff_groups.md`, every file with its ASSET-DIFF line).

| Mechanic group (rows in (a)) | Changed files | What changed (verified here unless marked) |
|---|---:|---|
| Melee weapons and attack interactions (5a-5c, 8d) | 54 | sword swings' knockback (5b), daggers' swing damage (5c), `DamageEntityParent` → `..._On_Hit`, 31 attacks re-parented to the hit interrupt (5a), item durability (8d); longsword and club changes are cosmetic (`ASSET-DIFF-0.7.2.md:485, 693, 700`) |
| Ranged weapons and projectile configs (6a-6c) | 46 | shortbow tier launch and damage (6a), crossbow combo and Big Arrow damage (6c), bow prototype configs, shortbow charge `StaminaRegenDelay` -1 steps (`ASSET-DIFF-0.7.2.md:252-264`); bomb change is cosmetic (`:478`) |
| Guard, bash and reload (5d) | 6 | guard roots lose `Interrupting: [Primary]`; shield bash regen delay Set -1.5 → Min -1; shortbow guard gains a signature-charge cost (`ASSET-DIFF-0.7.2.md:237, 1072-1127`) |
| Consumables (potions; 8a) | 5 | potion charge `FailOnDamage` true → false and walking speed 0.4 → 0.75 while drinking, cooldown variables, `MaxStack` 5 removed (`ASSET-DIFF-0.7.2.md:147-190, 351-384`) |
| NPC motion test roles (locomotion audit; 1b-1c, 10c) | 5 | memories metadata and translation keys only (`ASSET-DIFF-0.7.2.md:1136-1175`) |
| Player movement config (1a, 1d) | 2 | `MovementConfig/Default.json`: slope 30 and wall 15 added; `GameplayConfigs/Default.json`: ping removed, wilderness and world events added |
| Magic (7a) | 2 | wand cast particles only; bone staff loses a mana-50 cost override (`ASSET-DIFF-0.7.2.md:324, 942`) |
| Entity geometry and collision (row 12) | 2 | format only |
| Stats (8a) | 1 | `Stamina.json`: `UseIgnoreFlag` on two conditions. `Mana.json` (not referenced by the simulator) 0 → 100 (7a) |
| Blocks: break interaction (row 12, §3) | 1 | `Block_Break.json` `Failed.Next` now triggers `Trigger_Explosion_State_Generic` first |
| Crafting (11) | 1 | an interaction sound id only |
| NPC roles (perception/navigation audits) | 1 | `CAE_Hedera` action conditions |
| Audio (perception audit) | 1 | `AudioCategory` added |
| **Moved or missing** (8) | — | Trork and Kweebec role templates and two chase/attack instruction components moved (`Aggressive/`, `Neutral/`, `Steps/` → `Faction/`, `Instructions/Combat/`); `CAE_Goblin_Ogre.json` has no 0.7.2 counterpart (`ASSET-DIFF-0.7.2.md:1422-1434`) |

Not referenced by the simulator but relevant to the catalogue: `Server/Entity/Stats/Mana.json` (7a), the
hit-interrupt family (5a), 19 rune items (7b), `Deco_SpiderWeb.json` (3b), `MovementConfig/Mount.json`
(10e, format only).

## (d) The catalogue's 86 learned skills: what each needs and whether it can train on gen 7 today

**No skill can train on gen 7 today.** There is no 0.7.2 ruleset (only `rulesets/hytale_0_5_7/`), the
simulator labels every checkpoint 0.5.7 and `v0_7.py:18` accepts only 0.5.7 checkpoints (P1 not
started). The column below says what would still block each skill once P1 exists, using the row ids of
(a): **A** = a needed mechanic is ABSENT, **M** = a needed mechanic is MISMATCHED, **U** = the constructs
exist and the static values match, but no live 0.7.2 number exists yet. Row 3a (Region library) applies
only to skills trained in Regions; on flat arenas it drops out. Skills are listed in catalogue order
(86 rows of kind L, `SKILL_CATALOGUE.md` sections 1-13; extracted programmatically). The "gen 5" column is
the catalogue's own.

| Skill (§) | Mechanics needed (rows) | Catalogue gen-5 column | Gen-7 status (measured) |
|---|---|---|---|
| walk_to, run_to, sprint_to (§1) | 1a, 1b, 3a, 3b, 8a | sim (reach) | M: 3a, 3b. 1a has no native oracle |
| sneak_to (§1) | 1a, 3a | sim (sneak gait) | M: 3a. 1a no oracle |
| step_and_hop (§1) | 1a, 1c, 1f, 3a | sim (hop) | A: 1f (auto step-up). M: 1c, 3a |
| go_around (§1) | 1a, 1d, 3a | sim (detour) | A: 1d (slope/wall limits). M: 3a |
| slide (§1) | 1f | ? | A: 1f |
| land_safely (§1) | 1f; fall damage (not checked here) | ? | A: 1f |
| climb (§1) | 10a | none | A: 10a |
| swim_to (§1) | 10c | part (NPC swim only) | A: 10c |
| dive_and_breathe (§1) | 10c, 8b | ? | A: 10c. 8b U |
| glide (§1) | 10d, 8a | ? | A: 10d |
| double_jump (§1) | rocket boots (no construct: `double_jump`/`RocketBoots` 0 hits) | ? | A |
| dodge (§1) | 9, 8a | sim (4 directions) | U: 9 matches live (2 tests); 8a static only |
| hookshot_to (§1) | hookshot (no construct: 0 hits) | gen7 | A |
| ride (§1) | 10e | ? | A: 10e |
| bounce (§1) | bounce pads (no construct: 0 hits) | ? | A |
| follow_route (§1) | 1a, 3a; path tables | part (A* not wired) | M: 3a |
| look_at (§2) | 2a, 3a | sim | M: 3a in Regions; U on flat arenas |
| look_around (§2) | 2a, 2b, 3a | sim | M: 2b, 3a |
| search_for (§2) | 2b, 3a, 12 (perception) | part | M: 2b, 3a, 12 |
| track (§2) | 2b, 3a | part | M: 2b, 3a |
| spot_threats (§2) | 2b; NPC sight/hearing sensors (not checked) | part | M: 2b |
| mine_block, mine_in_direction, mine_down_stairs (§3) | 3a, 12 (mutable_blocks, inventory), 8d | part | M: 3a, 12 (0.7.2 also clears block health on any block write, `PORT-0.7.2.md`; not measured) |
| chop_tree, dig, pick_by_hand (§3) | as mine_block | part | M: 3a, 12 |
| shear_plants (§3) | as mine_block; shears (not checked) | ? | M: 3a, 12 |
| pick_up (§3) | 12 (inventory), 3a | part | M: 12, 3a |
| till, plant, water_crops, fertilize, harvest_crops (§4) | farming (no construct: `Tilled`, `Watering`, `Fertiliz`, `Crop_` 0 hits) | ? | A |
| lure_animal, milk_shear_collect, capture_creature (§4) | animal roles (the rulesets hold only Trork and Kweebec) | ? | A |
| place_block (§5) | 12 (mutable_blocks), 3a; 0.7.2 places through `PlaceModeSelect` → `Place_Drag` (`PORT-0.7.2.md`) | part (closed on the policy path) | M: 12, 3a |
| place_row (§5) | drag placement (no `Place_Drag` construct: 0 hits) | ? | A |
| build_up (§5) | 1c, place_block | ? | M: 1c, 12 |
| bridge, build_shape, fill_or_level (§5) | place_block, dig, 1a | ? | M: 12, 3a |
| carry_block (§5) | carry (0.7.2: carried container resolves interactions, `PORT-0.7.2.md`) | ? | not checked |
| place_furniture (§5) | place_block, 10b | ? | M: 12; 10b U |
| avoid_hazards (§8) | 3b, 3a; fall damage (not checked) | part | M: 3b, 3a |
| recover_stamina (§8) | 8a | sim | U: 8a (static match; bash cost agrees live) |
| hide (§8) | 2b, 1a | part | M: 2b |
| sword (§9) | 5b, 5d, 5e, 8c, 8d (17/23 and 23/23 sword rows); 5a inert vs Trork/Kweebec | sim | M: 5b, 5d, 8d |
| longsword (§9) | 5e, 8c; durability unchanged (0/17) | sim (legacy) | U |
| axe (§9) | 5e; durability unchanged (0/12) | sim (legacy) | U |
| battleaxe (§9) | 5a, 5d, 5e, 8d (11/15, 15/15) | sim | M: 5d, 8d |
| mace (§9) | 5a, 5d, 5e, 8d (7/12, 12/12) | sim (Groundslam world-conditional) | M: 5d, 8d |
| club (§9) | 5e; durability unchanged (0/22); scrap club loses an Ability3 reload entry | sim (legacy) | U |
| daggers (§9) | 5a, 5c, 5d, 8d (10/16, 16/16) | sim | M: 5c, 5d, 8d |
| spear (§9) | 5e; `Weapon_Spear_Leaf` unchanged; durability unchanged (0/6) | sim | U |
| claws (§9) | 5e; durability unchanged (0/1) | sim | U |
| shield_guard (§9) | 5d, 8a | sim | M: 5d |
| flamethrower (§9) | profile entry exists (`hytale_0_5_7.py`); not checked | ? | not checked |
| shortbow (§10) | 6a, 6b, 8d (9/13, 13/13) | sim | M: 6a, 8d |
| crossbow (§10) | 6b, 6c, 8d (2/2, 2/2) | sim | M: 6c, 8d |
| throw_spear (§10) | spear projectile | sim | MATCH: spear throw contact, open and wall (live) |
| throw_kunai (§10) | kunai charge bank | sim | M: 5f (roots one tick late per throw, live) |
| throw_bomb (§10) | bomb (0.7.2 change cosmetic) | sim | U |
| gun (§10) | `Gun_Blunderbuss_Bullet` (present in both zips, `Server/Projectiles/Player/Gun/`; the catalogue's "bullet id missing" did not hold for this file) | sim (gun kits) | U |
| darts_or_blowgun (§10) | 8d (blowgun 1/1, 1/1) | sim | M: 8d |
| fireball, wind_strike, chain_hook, ground_slam, charged_frost_shot (§11) | 7a, 7b | gen7 | A: 7b. M: 7a |
| staff_wand_spellbook (§11) | 7a (bone staff loses a mana-50 override) | part (scenario mana only) | M: 7a |
| guard, bash (§12) | 5d, 8a | sim | M: 5d |
| punish (§12) | 5e, 4 | part (contract, no collector) | U |
| keep_range, flank (§12) | 1a, 5e | part | U (1a no oracle) |
| kite (§12) | 1a, 6a, 6c | part | M: 6a, 6c |
| interrupt (§12) | 5a | gen7 | A: 5a; and no current opponent is eligible to be interrupted |
| finish (§12) | 1a, 3a, 4 | sim | M: 3a |
| follow, come_here, avoid, approach_friendly (§13) | 1a, 1b, 3a, 4 | part / sim | M: 3a |
| hide_from (§13) | 2b, 3a | part | M: 2b, 3a |
| signal (§13) | 2a, 3a | sim | M: 3a in Regions; U on flat arenas |

Counted by skill (86): **A 27, M 44, U 12, MATCH 1** (throw_spear), not checked 2 (carry_block,
flamethrower). Every A skill needs a new construct; every M skill needs recalibration first; no skill is
trainable on gen 7 until P1 lands.

## (e) What the simulator must change, in order, and the differential that gates each

This confirms `SIMULATOR-0.7.2.md` P2-P5 and extends it where the measurements found more ("new"
marks an item the plan does not list). Ordered for the first skills: locomotion first, then melee.

**Before P2 (part of P1): make the fidelity tools version-keyed.** On 0.7.2 several gates refuse before
measuring: `surrogate_crafting_native_compare.py:81-82`, `surrogate_door_native_compare.py:104-105`,
`surrogate_mutable_block_native_compare.py:97` and `surrogate_los_native_compare.py:88` require the 0.5.7
source audit or jar; `surrogate_swimming_native_compare.py:79` and `surrogate_locomotion_native_compare.py:62`
assert `native_server_version == "0.5.7"`; `adk/tools/native_motion_differential.py` defaults to
`--expected-server-version 0.5.9`. Every P2-P4 gate below depends on this (new).

**P2, locomotion and world**
1. **Choose the motion model the skills train on (new; owner decision).** Skills train with
   `motion_model="player"` (`types.py:549`; `arena/jax_env.py:199`), which no native differential can
   certify because the bridge body is an NPC on `MotionControllerWalk` (5.0 m/s live). Either train
   locomotion with `npc_walk` (certified on flat and ledge: 1.1444e-5) or obtain a player-client oracle
   (the locomotion audit's own contract: sprint `native_npc_oracle: cannot_certify_player_sprint_transition`,
   `surrogate_locomotion_source_audit.py:243-244`; swim `native_oracle: controlled_real_player_client_trace_required`,
   `:270`). Gate: the motion differential on the chosen model.
2. **Recapture the Region library on 0.7.2** and rebuild path tables, suites, biome table and horizons
   (plan item; confirmed: all three staged libraries are 0.5.7 captures). Gate: `capture_native_region.py`
   500/500 and `native_region_capture_compare.py` on 0.7.2.
3. **NPC jump (new).** Confirm, then port, the native dt handling (gravity appears to use the previous
   tick's duration) and fix the landing velocity that already failed on 0.5.9. Gate: `native_motion_differential.py` with
   `--jump-steps` (in place 12; single 30; jump + forward 30) <= 5e-4, including runs with long ticks.
4. **Line of sight through partial hitboxes** (plan item; confirmed in the 0.7.2 jar). Gate: a new bridge
   fixture with a slab, stair or fence in the target ray, compared like `native_geometry_compare.py`'s three
   current rays (which pass).
5. **Slope and wall limits (30°, 15°) and walking descent** (plan item). Gate: a new stairs/slope
   fixture and the motion differential.
6. **Reset alignment** (plan item, refined): document the kill_trork first step (+1 world tick); the NPC
   activation pin (44/46) holds on 0.7.2. Gate: `test_active_combat_compare.py --root-tick-samples` and
   `R/scripts/reset_timing_probe.py`.

**P3, combat content**
1. **Melee values** (plan item, extended): knockback of all three sword swings, not only Swing_Left
   (Right 7 → 5.0, Down 12 → 10.8, Y 1 → 0.2); daggers +50 %; durability regenerated from 0.7.2 (57 maximum
   and 82 loss rows of 143); the guard rule (four guard roots lose `Interrupting: [Primary]`, thrust gains
   `InterruptedBy: [Secondary]`, shield bash regen delay Set → Min) (new). Gate:
   `test_jax_native_combat_compare.py` sword and shield-bash tests, a knockback displacement differential
   (TRAINING's 0.19 m), the active-combat compare (it needs `SimulatorBridgeMain` on port 5557 as well).
2. **Hit interrupt, stun and flinch** (plan item, scoped): needed only against NPC families that carry
   `Component_Instruction_Hit_Interrupt` (Template_Intelligent, Template_Predator, goblins, skeletons,
   spiders, void, flying aggressive), not for the current Trork and Kweebec duels. Gate: a new
   interrupt/flinch differential against an eligible NPC (flinch 0.5 s, stun 0.7 s, attack cancel).
3. **Ranged** (plan item, extended): shortbow tier launch speeds 12 / 22 / 35 / 50 as well as damage;
   crossbow damage. Gate: `test_native_shortbow_outer_root_strengths_match_jax`, the crossbow tests in
   `tests/fidelity/combat/projectiles/`, a range differential (<= 2 % per tier).
4. **Mana and runes**: mana 100, +1 per 0.3 s (plan); runes are new constructs (19 items; not in the
   plan). Gate: a mana resource differential and one differential per rune.
5. **Dodge head**: reduce the 4 dodge slots to 2 (forward and back are already masked). Gate: the dodge
   tests (both pass on 0.7.2). Note for `sim/gen7/GEN7-PLAN.md` section 3, which lists a "two-direction
   dodge" construct for 0.7.2: the 0.5.10 dodge assets are identical (left and right only) and the
   simulator already authors only left and right (`mechanics/schema/contract.py:81`); only the head size
   differs.

**P4, crafting** (plan item, extended): capacity 2,047; packer support for excluded-material inputs (988);
bench-id normalisation (22 live ids against the catalogue's 19). Gate: the crafting native compare once its
0.5.7 gate is profile-keyed; the crafting source audit at a 0.7.2 baseline.

**P5, source pins** (plan item, extended): 11 new baselines, and code edits first for the four audits that
crash (removed classes; role assets moved to `Faction/` and `Instructions/Combat/`). Gate: 11/11 on 0.7.2.

**Not in P2-P5 (new mechanics):** climbing, player swimming, gliding, riding, slide, landing roll,
automatic step-up, double jump, hookshot, bounce pads, farming, drag placement. All are absent; the
player-only ones have no server-side oracle while the bridge body is an NPC.

## (f) Risks, and what could not be measured

- **Not measured live, with the reason:** player movement (no oracle, 1a); slopes and stairs (no
  fixture, 1d-1e); line of sight through partial hitboxes (no fixture, 2b); Region recapture (not run;
  TRAINING's 500/500 and 46/250 are cited, not re-measured, 3a); knockback displacement (the
  active-combat compare also needs a `SimulatorBridgeMain` listener on 5557, 5b); mana, runes, oxygen
  (no differential exists, 7a, 7b, 8b); swimming and the walk-heading compare (they assert server 0.5.7);
  doors (the compare refuses on a non-0.5.7 source audit); the hit interrupt on an eligible NPC (5a).
- **Machine load.** Other sessions kept the load average at 6-10; long world ticks (up to 0.0352 s)
  made the jump runs fail earlier than they would on a quiet machine, so the 0.7.2 jump error sizes (1c)
  are not comparable with the 0.5.9 ones.
- **Static inferences.** Row 5a's "current opponents are not interruptible" comes from walking role
  references with a regular expression (positive controls Skeleton_Fighter and Goblin_Scrapper do reach
  the component); row 2b's rule comes from the 0.7.2 constant pool and `PORT-0.7.2.md`, not from a ray.
- **The claim most likely to be wrong:** row 1c's mechanism (native jump gravity uses the previous
  tick's duration). It rests on the per-step velocity deltas of one jump-in-place run (its long tick at
  state 4 and the tick after), plus the first failing state coinciding with the first long tick after
  launch in all six jump runs on 0.7.2. No run with only nominal ticks could be obtained under this load,
  so another jitter-sensitive jump integration is not excluded. Next most
  likely: row 5a's inference that Trork and Kweebec cannot be interrupted (static reading, no live hit
  test), and the cause of the no-hit ranged runs (6a, 6b), which this run did not isolate.
- **Control build.** The installed 0.5 is 0.5.10, so a 0.5.7 → 0.5.10 change cannot be told apart from
  "unchanged" here (the eleven pinned class sets are byte-identical between them).
- **Side effects of this run.** Native-evidence lease files were redirected to `R/lease/` (the stale
  `HytaleRL/.codex-local/...-5556.json` was not touched) by an in-process patch
  (`R/scripts/lease_redirect.py`); the audit driver's import rewrote one git-ignored bytecode file,
  `HytaleRL/hytalegym/tests/fidelity/__pycache__/surrogate_crafting_source_audit.cpython-312.pyc`, before
  `PYTHONDONTWRITEBYTECODE=1` was set. No tracked file was modified.

## Receipts

All under `R/` = `/Users/arush/.claude/jobs/31947bd7/tmp/fidelity072/`.

| Receipt | What |
|---|---|
| `00_inputs.txt` | jar manifests and hashes, bridge hash, assets sizes, decompile version check |
| `01_source_audits.log`, `01_source_audit_table.md`, `audits/{072,0510}/` | the eleven audits on 0.7.2 and 0.5.10 |
| `02_locomotion_constants_vs_assets.md` | `types.py` constants against both `MovementConfig` and `Stamina` files |
| `02_weapon_asset_delta.txt`, `02_projectile_config_delta.txt`, `02_durability_delta.txt` | weapon damage, knockback, launch, durability deltas |
| `02_static_mechanics_receipts.txt` | dodge, hit interrupt graph, stats, signature cost |
| `02_asset_diff_groups.md` | ASSET-DIFF changed files by mechanic |
| `javap/PositionCache_{0510,072}.txt` | 0.5.10 and 0.7.2 `PositionCache` bytecode |
| `live/launcher.log`, `live/run1/server.log` | the live 0.7.2 server |
| `live/03a_smoke.json.txt` | `smoke_bridge_072.py`: 11/11 checks |
| `live/03b_motion.log`, `live/motion/*.json` | motion differentials |
| `live/03c_geometry.log` | `native_geometry_compare.py` |
| `live/03d_crafting_compare.out`, `live/03e_crafting_capture.json`, `live/03e_crafting_manifest_072.json` | crafting |
| `live/03f_reset_timing*.json`, `live/03g_root_ticks.{log,json}` | reset timing and NPC activation |
| `live/03h_pytest_combat.{log,xml}` | `tests/fidelity/combat` native tests |
| `live/03i_pytest_selected.{log,xml}`, `live/03j_door_compare.out` | selected combat compares, door compare |
| `scripts/` | every script used |

## (g) Phase 3 on the gen-7 render (2026-10-18; `sim/gen7` overlay)

Render: `hytalegym/sim/gen7/engine`, rendered from commit 8707eec (its `jax/`, `world/`, `geometry/`
and `rulesets/` are unchanged to HEAD 97539bf). Bridge: 0.7.2 jar SHA-256 686BC4BC...0012 on
127.0.0.1:5556, started by `tools/launch_bridge_072.py` after the native lease was acquired and
released (the port was free); no other process was stopped. Receipts (JSON, command, inputs, numbers):
`~/.claude/jobs/31947bd7/tmp/phase3/receipts/` (`r2_live_gen7_motion_and_geometry.json`,
`r3_los_probe_full_cubes.json`, `r4_jump_control.json`, `locomotion_rows_head_vs_render.json`).

| Row | Gen-7 differential (what ran) | Measured | Bar | Status |
|---|---|---|---|---|
| 1b | motion, ledge fixture (flat floor and three-block trench), 30 steps, JAX = gen-7 render | max position error 1.1444e-05 m; velocity 3.4e-04 | <= 5e-4 m | MATCH on the gen-7 render |
| 2a | native_geometry_compare probes on the gen-7 render with hook 2b active: LOS wall, diagonal graze, diagonal block | 3 of 3 rays agree with native under both rules (gen-5 rule and detail rule) | every fixture ray | MATCH (full cubes; partial blocks not measured, see 2b) |
| 1c | NPC jump: one jump in place over 12 steps; single jump with forward over 30 steps | gen 5 2.384 m and 2.415 m; gen 7 2.418 m; the JAX side applies 11.8 (player impulse), native 5.099 | <= 5e-4 m | MISMATCH, unchanged. Cause: committed `jax/combat/types.py` gives the NPC learner `PLAYER_JUMP_FORCE` on purpose; the uncommitted working tree uses 5.099. The earlier 0.0044 m receipt was measured on that working tree |
| 1d | slope fixture, motion differential | not run: no slope fixture on the bridge; the 0.7.2 server jar reads neither key (scan of 38,436 classes: `MovementConfig` and the client struct `MovementSettings` only) | <= 5e-4 m | UNMEASURED |
| 2b | partial-box LOS ray (slab, stair or fence in the ray) | not run: `half_block` runs only under `native_fidelity`, which has no target and no `target_los` | every fixture ray | UNMEASURED (construct implemented, `sim/gen7/mechanics/line_of_sight.py`) |
| 1e | walk descent (minimum descent rate) | not run; the premise (TRAINING-0.7.2.md item 8, "(code)") was not located in the walk controller's fall-speed method | <= 5e-4 m | UNMEASURED; switch not hooked |

Also measured on the same bridge: the native half-block probe (`HALF_BLOCK_DIFFERENTIAL_PASS`, its own
2e-3 m tolerance, not the 5e-4 bar) and the ceiling and unloaded-chunk probes pass on the gen-7 render.

Locomotion audit at the 0.7.2 baseline (`tests/fidelity/surrogate_locomotion_source_audit.py`, 2026-10-18):
0.7.2 semantic SHA-256 `db3dc468...`, server `23f71f03...` (matches the ground truth above);
`matches_0_5_7_baseline` False: 14 class files drift (among them `MotionControllerWalk`, `MovementConfig`,
`PhysicsMath`) and 6 assets drift (`MovementConfig/Default.json`, `GameplayConfigs/Default.json`). The
0.5.10 control matches its 0.5.7 pin (`7f7ce39d...`). The semantic hash covers only those hashes and the
script's hard-coded `SOURCE_CONTRACT`, so it does not verify the 0.7.2 locomotion semantics. Baseline
recorded here; the pins are not changed (P5).

Gaps that remain: row 1c (committed jump impulse; an engine change for every generation, so an owner
decision with these numbers), rows 1d, 1e and 2b (no fixture or no reader), and the production
perception line of sight (`jax/world/perception/los.py`), which hook 2b does not cover.

## (h) Phase 3b and phase 4 on the gen-7 render (2026-10-18; render pinned to 8707eec)

Receipts: `~/.claude/jobs/31947bd7/tmp/phase4/` (`receipts/`, `jump/`, `los/`, `javap/`). Bridge: 0.7.2 jar 686BC4BC...0012 on 127.0.0.1:5556 (unchanged jar and pin). Only what ran is claimed.

| Row | Differential (what ran) | Measured | Bar | Status |
|---|---|---|---|---|
| 1c | `adk/tools/native_motion_differential.py --jump-steps 1 --steps 12 --forward-steps 0`, gen-7 render, live | first jump step native = JAX (vy 5.09902; y 65.17525 both); max position error 0.02460 m | <= 5e-4 m | MISMATCH (first step matches; residual is F1c) |
| 1c | same tool, `--steps 30 --jump-steps 1` (jump, then forward) | max position error 0.97770 m | <= 5e-4 m | MISMATCH |
| 1c | arithmetic on both receipts (airborne steps) | native dv = -32 x previous dt, deviation <= 2.4e-4 m/s over 16 steps; current-dt deviation 0.01 to 0.21 m/s | - | cause F1c, confirmed by receipt arithmetic (no code experiment) |
| 1d | jar scan of 38,436 classes (`slope_wall_key_scan_0722_jar.json`) | no 0.7.2 reader of the two keys; gen 7 `slope_limits` False | - | DECIDED no-op; no slope motion measured |
| 1e | javap of `MotionControllerWalk.postReadPosition`: descend move clamped by `vy = min(vy, -fallSpeed * dt)` | code support for the premise | - | UNMEASURED (no stairs or slope fixture); not hooked |
| 2b | captured live frames (`los_wall`, `los_diagonal_graze`, `los_diagonal_block`, seed 570057): world walk and production perception walk, gen-5 identity and gen-7 render | 3 of 3 match native `target_los` on both walks and both renders | every fixture ray (full cubes) | MATCH (full cubes) |
| 2b | implementation check on the captured `half_block` frame (`Rock_Stone_Half` at (0,65,-2), bottom detail box) | ray over the box: visible under the rule, blocked under gen 5; ray through the box: blocked under both | no native oracle (not fidelity) | UNMEASURED (partial box, native) |
| 5c | `tests/fidelity/test_jax_native_combat_compare._native_archetype_breadth_report`, all 15 scalar dagger profiles, gen-7 render, seeds 20260729 and the control seed (first contact damage) | native = JAX on both seeds for all 15 (bronze 5, mithril 12, crude 3, adamantite 9, ...) | damage exact | MATCH on the render, 15 of 15 (Iron not in this run) |
| 5c | gen-5 identity render, bronze and mithril (control) | native 5 vs JAX 3 (bronze); native 12 vs JAX 8 (mithril) | damage exact | MISMATCH on gen 5 (the change is needed) |
| 5c | `Weapon_Daggers_Void_Crystal` in the 0.7.2 table, no simulator variant row | - | - | ABSENT (not modelled) |

**Contract note (the "learner-v3 actor-evidence contract mismatch" abort; cause corrected and fixed 2026-10-18).** The bridge constant `CONTRACT_SHA256` (D47E87B2) is the host hash when the rendered token module resolves `_WORLDGEN_ARTIFACT_ROOT` to `HytaleRL/artifacts/worldgen`, which holds the staged Region library (gitignored, `.gitignore:136`). The relocation rule had anchored that climb on `hytalegym.__file__`. Under the repo-root shim (`cd HytaleRL && python -m pytest ...`) that file is `HytaleRL/hytalegym/__init__.py`, one level too high, so the path became `/Users/arush/hydl/artifacts/...` (absent), the host hash became B133A81, and the bridge refused the host (the 25 aborts). The rule now anchors on `hytalegym.envs.__file__` (`sim/common/render.py`; test `tests/sim/test_render_anchors.py` checks both import modes); the shim probe gives D47E87B2 again. No contract-hashing code was changed. A clean clone without the gitignored artifact still gets B133A81, so native compares need the artifact present in the tree.

**Native compares after the fix (2026-10-18; gen-7 render, run from `cd HytaleRL` through the shim, bridge 5556, jar 686BC4BC...0012).** 23 selected tests (the 14 dynamic learner rows, the queue drain, three dodge tests: lateral, alternating lateral at two seeds, policy boundary; the shield bash; the three learner-v3 tests): 0 contract-mismatch refusals (test output and bridge log), 8 passed, 15 failed as measurements. Fourteen dynamic learner rows report raw observation mismatches (width 8272; 107 to 109 features; maximum error 1.0; world, resource and guard groups). `test_native_learner_v3_full_row_matches_source_aligned_jax` reports three float32 entries (maximum error 7.45e-9) against an exact-zero assertion. Stock-engine attribution is not run here. Identity gate (gen 5, pinned to 8707eec because HEAD does not import): 4 passed (36:34). Receipts: `~/.claude/jobs/31947bd7/tmp/phase4/receipts/native_compares_gen7.log`, `final2_identity_pinned_8707eec.txt`.

**Still open:** row 2b partial boxes (the bottom-box `Rock_Stone_Half` cannot be placed to separate the rules from the spawn geometry; needs a rotation API or an eye offset); row 1d slope motion; row 1e descent; row 1c residual (F1c, an engine-wide change needing the owner's decision); Part B constructs (hit interrupt, knockback model, mana, hookshot, dodge head, projectile tiers) not started.
