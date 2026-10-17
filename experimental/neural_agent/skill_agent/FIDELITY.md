# Gen-7 fidelity ledger (append-only)

This ledger records every native differential on the live Hytale 0.7.2 server that calibrates a simulator
mechanic, with its receipt. A verb's training result counts only after the mechanics it depends on have
passed their rows in BUILD.md section 4. Entries are appended, never edited or deleted; a later entry
supersedes an earlier one for the same mechanic. Created 2026-10-17 with every seed row `pending`; nothing
has passed.

## Seed rows (2026-10-17)

Each subsection copies one row of BUILD.md section 4, in table order. `mechanic:` is the mechanic cell with
its catalogue numbers removed; those numbers are in `expected:`. `differential:` and `pass bar:` are the
Verification and Pass bar cells, verbatim.

### 1. Movement
- mechanic: Movement: walk, run, sprint, sneak, jump, slide, landing roll, step-up
- expected: walk 1.65; run 5.5; sprint 7.0; sneak 3.0; jump 11.8
- differential: locomotion source audit on 0.7.2 assets and jar; flat and slope motion differential
- pass bar: position error <= 5e-4 per step (P2 gate)
- status: pending
- receipt: none

### 2. Line of sight
- mechanic: Line of sight (0.7.2 passes non-transparent blocks unless a detail hitbox is hit)
- expected: none (no catalogue number in the mechanic)
- differential: LOS differential on an occlusion fixture; the catch rule and the `hidden` opening updated
- pass bar: agreement with native on every fixture ray
- status: pending
- receipt: none

### 3. Terrain
- mechanic: Terrain: renumbered block ids, changed Region sections, web speed
- expected: none (no catalogue number in the mechanic)
- differential: Region library recaptured on 0.7.2; path tables, suites, biome table, horizons rebuilt; audit
  worlds stay unseen
- pass bar: 500/500 sections captured; suites regenerated
- status: pending
- receipt: none

### 4. Reset timing
- mechanic: Reset timing (first step after reset)
- expected: 34-68 ticks
- differential: native episode alignment check
- pass bar: documented offset, applied in the session clock
- status: pending
- receipt: none

### 5. Melee
- mechanic: Melee: hit interrupt, stun and flinch on weapon attacks (new construct), sword knockback, dagger
- expected: 0.5 s flinch; 31 weapon attacks; sword knockback 4.3 at (0, 0.2, -1); dagger +50 %
- differential: new simulator construct; combat compare and active-combat compare re-run with 0.7.2 receipts;
  interrupt/flinch differential
- pass bar: per-hit displacement and flinch timing match native within tolerance
- status: pending
- receipt: none

### 6. Ranged
- mechanic: Ranged: damage cut; bow draw tiers; launch and gravity per projectile (bow full draw, crossbow,
  Big Arrow)
- expected: damage cut 3-4x; bow full draw 85 / 25; crossbow 40 / 10; Big Arrow 100 / 40
- differential: projectile differential: launch speed, drop, hit distance per tier
- pass bar: range error <= 2 % per tier
- status: pending
- receipt: none

### 7. Mana and runes
- mechanic: Mana; runes (fireball, straight flight; wind strike; chain hook; ground slam; charged frost
  tiers)
- expected: Mana 100, +1 per 0.3 s; fireball 30 mana, 2 s cooldown, 0.87 s cast; chain hook to 1.5; ground
  slam radius 3
- differential: mana construct; rune differentials per rune
- pass bar: cost, cooldown, cast time, flight and effect match native
- status: pending
- receipt: none

### 8. Stamina, oxygen, signature energy, durability tables
- mechanic: Stamina (sprint, guard, dodge), oxygen, signature energy, durability tables
- expected: Stamina 10; oxygen 100 (-6/s submerged)
- differential: resource differentials
- pass bar: rates match native within 5 %
- status: pending
- receipt: none

### 9. Dodge
- mechanic: Dodge: left and right only, impulse, invulnerable, stamina
- expected: impulse 13; 0.25 s invulnerable; 2 stamina
- differential: dodge differential; the locomotion head's four dodges reduced to two
- pass bar: impulse and timing match
- status: pending
- receipt: none

### 10. Climbing, doors, swimming, gliding, riding
- mechanic: Climbing (climbable blocks), doors, swimming, gliding, riding
- expected: 44 climbable blocks
- differential: new mechanics in the simulator, each with a differential
- pass bar: present and matching, or the verb waits
- status: pending
- receipt: none

### 11. Crafting
- mechanic: Crafting: recipes, benches, cook
- expected: 2,047 recipes; 19 benches; cook 2 s
- differential: crafting native compare on 0.7.2
- pass bar: catalogue and timings match
- status: pending
- receipt: none

### 12. Source-pin audits
- mechanic: The source-pin audits (perception, locomotion, navigation, doors, mutable blocks, explosion
  mutation and dynamics, crafting, actor world state, inventory, entity geometry)
- expected: 11 source-pin audits
- differential: re-recorded from the 0.7.2 jar and assets
- pass bar: all 11 pass on 0.7.2 (P5 gate)
- status: pending
- receipt: none

### 13. Contracts and checkpoints
- mechanic: Contracts and checkpoints: new ruleset bytes change the contract stamp
- expected: none (no catalogue number in the mechanic)
- differential: every checkpoint retrained; native transfer `execution_valid` with one skill checkpoint on
  0.7.2 (P6 gate)
- pass bar: one skill transfers and acts validly on the live server
- status: pending
- receipt: none

### 14. Legal observation
- mechanic: Legal observation
- expected: none (no catalogue number in the mechanic)
- differential: the opt-in legal mode on for every result that counts; the leakage three-world test
- pass bar: no privileged field reaches a body
- status: pending
- receipt: none

## How to append

Append each differential as a new entry at the end of this file, using the template below. Never edit a seed
row or an earlier entry. Use the mechanic name from the seed heading, for example `Melee`, as `<mechanic>`.
Write `pass` only when the row's pass bar is met on the receipt the entry names. `<date>` is YYYY-MM-DD.

```
## <date> — <mechanic>: <pass | fail | pending>
- command: <the exact command run, with its flags>
- server: <Hytale server version>; bridge jar sha256 <hex>
- receipt: <path to the recorded evidence>
- result: <the measured numbers, set against the pass bar>
- verdict: <pass | fail | pending>, with the reason
```

## 2026-10-17 — Movement: fail
- command: the gen-7 fidelity check by another session (live 0.7.2 server, 2026-10-17 21:22-22:30 UTC; `tools/launch_bridge_072.py --bridge-port 5556 --game-port 25572`, then the motion, geometry, reset, crafting and combat differentials named in the report); this entry points at its table (a) rows 1a-1f
- server: Hytale 0.7.2 (HytaleServer.jar revision d769dc3d, SHA-256 23f71f03...0aff); bridge jar sha256 686BC4BC3DAE481AE857875961DAE75C02E2F4102B75B9B4193E60246EF80012
- receipt: `HytaleRL/hytale-plugin-0.7.2/FIDELITY-0.7.2.md` (commit cdeace2), table (a) rows 1a-1f; raw receipts in that report's `R/` folder (outside the repository)
- result: NPC walk passes: 30 steps flat, max position error 1.1444e-5 m (bar 5e-4), the 3-block ledge walk 1.85e-5 m. NPC jump fails: 0.703 m single jump with forward motion, 0.304 m jump with forward, 4.40e-3 m in place; the divergence starts on the first world tick longer than 1/30 s (native gravity uses the previous tick's duration; the report's inference). Slope and wall limits (30 and 15 degrees, new in 0.7.2) are absent. The player model the skills train with (5.5 m/s) cannot be measured through the bridge, whose body is an NPC on MotionControllerWalk (5.0 m/s); its 21 constants equal the 0.7.2 MovementConfig statically
- verdict: fail: the pass bar holds for the NPC flat walk only; jump, slopes and the player model fail or are unmeasured

## 2026-10-17 — Line of sight: fail
- command: the gen-7 fidelity check by another session (live 0.7.2 server, 2026-10-17 21:22-22:30 UTC; `tools/launch_bridge_072.py --bridge-port 5556 --game-port 25572`, then the motion, geometry, reset, crafting and combat differentials named in the report); this entry points at its table (a) rows 2a-2b
- server: Hytale 0.7.2 (HytaleServer.jar revision d769dc3d, SHA-256 23f71f03...0aff); bridge jar sha256 686BC4BC3DAE481AE857875961DAE75C02E2F4102B75B9B4193E60246EF80012
- receipt: `HytaleRL/hytale-plugin-0.7.2/FIDELITY-0.7.2.md` (commit cdeace2), table (a) rows 2a-2b; raw receipts in that report's `R/` folder (outside the repository)
- result: full-cube occluders: 3/3 fixture rays agree on 0.7.2 (wall blocked, diagonal graze visible, diagonal block blocked; every geometry differential in the receipt passes). The 0.7.2 rule (an opaque block occludes only where a detail hitbox is hit) is not ported: the simulator still occludes the whole cell, and no bridge fixture puts a partial-hitbox block in a ray
- verdict: fail: the mechanic the row names, the 0.7.2 rule, is not implemented (static mismatch); the existing fixture rays agree

## 2026-10-17 — Terrain: fail
- command: the gen-7 fidelity check by another session (live 0.7.2 server, 2026-10-17 21:22-22:30 UTC; `tools/launch_bridge_072.py --bridge-port 5556 --game-port 25572`, then the motion, geometry, reset, crafting and combat differentials named in the report); this entry points at its table (a) rows 3a-3b
- server: Hytale 0.7.2 (HytaleServer.jar revision d769dc3d, SHA-256 23f71f03...0aff); bridge jar sha256 686BC4BC3DAE481AE857875961DAE75C02E2F4102B75B9B4193E60246EF80012
- receipt: `HytaleRL/hytale-plugin-0.7.2/FIDELITY-0.7.2.md` (commit cdeace2), table (a) rows 3a-3b; raw receipts in that report's `R/` folder (outside the repository)
- result: every staged Region library was captured on server 0.5.7 (manifests' capture_contract.server_version); not recaptured in this run; 0.7.2 web movement values differ statically (horizontal 0.65 -> 0.45, jump 0.5 -> 0.45, drag 0.45 -> 0.82)
- verdict: fail: no 0.7.2 recapture yet (bar: 500/500 sections, suites and horizons rebuilt)

## 2026-10-17 — Reset timing: pending
- command: the gen-7 fidelity check by another session (live 0.7.2 server, 2026-10-17 21:22-22:30 UTC; `tools/launch_bridge_072.py --bridge-port 5556 --game-port 25572`, then the motion, geometry, reset, crafting and combat differentials named in the report); this entry points at its table (a) rows 4
- server: Hytale 0.7.2 (HytaleServer.jar revision d769dc3d, SHA-256 23f71f03...0aff); bridge jar sha256 686BC4BC3DAE481AE857875961DAE75C02E2F4102B75B9B4193E60246EF80012
- receipt: `HytaleRL/hytale-plugin-0.7.2/FIDELITY-0.7.2.md` (commit cdeace2), table (a) rows 4; raw receipts in that report's `R/` folder (outside the repository)
- result: Trork first turn 44 and first root 46 in 5/5 samples over 2 seeds, equal to the pins. The first step after reset executes 4 engine ticks (1 in the motion harness), with one extra world tick in kill_trork sessions. The "34-68 ticks" figure did not reproduce
- verdict: pending: the offset is now documented; it is not yet applied in the session clock

## 2026-10-17 — Melee: fail
- command: the gen-7 fidelity check by another session (live 0.7.2 server, 2026-10-17 21:22-22:30 UTC; `tools/launch_bridge_072.py --bridge-port 5556 --game-port 25572`, then the motion, geometry, reset, crafting and combat differentials named in the report); this entry points at its table (a) rows 5a-5f
- server: Hytale 0.7.2 (HytaleServer.jar revision d769dc3d, SHA-256 23f71f03...0aff); bridge jar sha256 686BC4BC3DAE481AE857875961DAE75C02E2F4102B75B9B4193E60246EF80012
- receipt: `HytaleRL/hytale-plugin-0.7.2/FIDELITY-0.7.2.md` (commit cdeace2), table (a) rows 5a-5f; raw receipts in that report's `R/` folder (outside the repository)
- result: match: Trork battleaxe hit ticks [16, 16, 15, 16, 15] in 5/5 samples, and seeded contacts of a cobalt battleaxe, a scrap NPC mace and a thorium sword (damage and hit tick). Mismatch: dagger damage 5.0 native vs 3.0 simulator; sword knockback values stale (0.7.2: 4.3 / 5.0 / 10.8 at (0, 0.2, -1 or -2)); the agent's own hits land 1-2 ticks later on 0.7.2 (iron sword tick 9 vs 7, kunai roots 11, 22, 33 vs 10, 20, 30); the guard attack-cancel rule is the 0.5.7 one (static). The hit interrupt, stun and flinch construct is absent; by the 0.7.2 role graph it does not reach Trork_Brawler, Trork_Unarmed or Kweebec_Razorleaf (the report's inference)
- verdict: fail: damage, knockback and dispatch timing differ from 0.7.2

## 2026-10-17 — Ranged: fail
- command: the gen-7 fidelity check by another session (live 0.7.2 server, 2026-10-17 21:22-22:30 UTC; `tools/launch_bridge_072.py --bridge-port 5556 --game-port 25572`, then the motion, geometry, reset, crafting and combat differentials named in the report); this entry points at its table (a) rows 6a-6c
- server: Hytale 0.7.2 (HytaleServer.jar revision d769dc3d, SHA-256 23f71f03...0aff); bridge jar sha256 686BC4BC3DAE481AE857875961DAE75C02E2F4102B75B9B4193E60246EF80012
- receipt: `HytaleRL/hytale-plugin-0.7.2/FIDELITY-0.7.2.md` (commit cdeace2), table (a) rows 6a-6c; raw receipts in that report's `R/` folder (outside the repository)
- result: shortbow tier 0 damage 1.0 native vs 6.0 simulator; tiers 0-3 launch speeds stale (0.7.2: 12 / 22 / 35 / 50 vs 15 / 30 / 50 / 70) and damage stale (1 / 2 / 4 / 7 vs 6 / 9 / 12 / 16); in 10 shortbow runs the native arrow registered no hit where the simulator predicts one (cause not isolated). Full draw 85 / 25, crossbow 40 / 10 and Big Arrow 100 / 40 launch values match statically; crossbow damage stale (iron combo 27 -> 13)
- verdict: fail: tier values and damage differ; the live contacts are inconclusive

## 2026-10-17 — Mana and runes: fail
- command: the gen-7 fidelity check by another session (live 0.7.2 server, 2026-10-17 21:22-22:30 UTC; `tools/launch_bridge_072.py --bridge-port 5556 --game-port 25572`, then the motion, geometry, reset, crafting and combat differentials named in the report); this entry points at its table (a) rows 7a-7b
- server: Hytale 0.7.2 (HytaleServer.jar revision d769dc3d, SHA-256 23f71f03...0aff); bridge jar sha256 686BC4BC3DAE481AE857875961DAE75C02E2F4102B75B9B4193E60246EF80012
- receipt: `HytaleRL/hytale-plugin-0.7.2/FIDELITY-0.7.2.md` (commit cdeace2), table (a) rows 7a-7b; raw receipts in that report's `R/` folder (outside the repository)
- result: mana maximum is 0 by default in the simulator (0.7.2: 100 / 100, +1 per 0.3 s); no rune exists in the simulator (0.7.2 has 7 ability runes, Rune_Empty and 11 modifier runes)
- verdict: fail: the construct is absent

## 2026-10-17 — Stamina, oxygen, signature energy, durability tables: fail
- command: the gen-7 fidelity check by another session (live 0.7.2 server, 2026-10-17 21:22-22:30 UTC; `tools/launch_bridge_072.py --bridge-port 5556 --game-port 25572`, then the motion, geometry, reset, crafting and combat differentials named in the report); this entry points at its table (a) rows 8a-8d
- server: Hytale 0.7.2 (HytaleServer.jar revision d769dc3d, SHA-256 23f71f03...0aff); bridge jar sha256 686BC4BC3DAE481AE857875961DAE75C02E2F4102B75B9B4193E60246EF80012
- receipt: `HytaleRL/hytale-plugin-0.7.2/FIDELITY-0.7.2.md` (commit cdeace2), table (a) rows 8a-8d; raw receipts in that report's `R/` folder (outside the repository)
- result: durability tables are stale: of 143 weapon rows, 57 differ in maximum and 82 in loss per hit on 0.7.2 (iron sword 250, 0.13 per hit vs 120, 0.21). Stamina (10, -4, +0.3 per 0.1 s), oxygen (100, +25 / -3 per 0.5 s) and signature energy (Vortexstrike 100 % of the bar) match statically and were not measured live
- verdict: fail: durability fails; the other three are unmeasured

## 2026-10-17 — Dodge: pass
- command: the gen-7 fidelity check by another session (live 0.7.2 server, 2026-10-17 21:22-22:30 UTC; `tools/launch_bridge_072.py --bridge-port 5556 --game-port 25572`, then the motion, geometry, reset, crafting and combat differentials named in the report); this entry points at its table (a) rows 9
- server: Hytale 0.7.2 (HytaleServer.jar revision d769dc3d, SHA-256 23f71f03...0aff); bridge jar sha256 686BC4BC3DAE481AE857875961DAE75C02E2F4102B75B9B4193E60246EF80012
- receipt: `HytaleRL/hytale-plugin-0.7.2/FIDELITY-0.7.2.md` (commit cdeace2), table (a) rows 9; raw receipts in that report's `R/` folder (outside the repository)
- result: test_native_policy_dodge_boundary_matches_advertised_capability (4.5 s) and test_native_lateral_dodge_projects_mechanics_when_supported (61.1 s) pass on 0.7.2: left and right only, force 13, 0.25 s invulnerable, 2 stamina, regeneration delay -0.7 (all 7 dodge assets identical to 0.5.10). test_live_lateral_dodge_blocks_overlapping_native_hit did not measure the dodge (its control attack never landed). The gen-5 policy head still carries four dodges; skill_agent's gen-7 layout disables them and dodges through the dodge head
- verdict: pass: impulse and timing match on the two passing differentials (receipts read in this session)

## 2026-10-17 — Climbing, doors, swimming, gliding, riding: fail
- command: the gen-7 fidelity check by another session (live 0.7.2 server, 2026-10-17 21:22-22:30 UTC; `tools/launch_bridge_072.py --bridge-port 5556 --game-port 25572`, then the motion, geometry, reset, crafting and combat differentials named in the report); this entry points at its table (a) rows 10a-10e
- server: Hytale 0.7.2 (HytaleServer.jar revision d769dc3d, SHA-256 23f71f03...0aff); bridge jar sha256 686BC4BC3DAE481AE857875961DAE75C02E2F4102B75B9B4193E60246EF80012
- receipt: `HytaleRL/hytale-plugin-0.7.2/FIDELITY-0.7.2.md` (commit cdeace2), table (a) rows 10a-10e; raw receipts in that report's `R/` folder (outside the repository)
- result: climbing, player swimming, gliding and riding are absent from the simulator; doors are unmeasured: the shipped door compare stops inside its source audit on 0.7.2 (KeyError for BlockAccessor.class, a class 0.7.2 removed); the swimming compare asserts native_server_version == "0.5.7" and was not run
- verdict: fail: four mechanics absent, one unmeasured

## 2026-10-17 — Crafting: fail
- command: the gen-7 fidelity check by another session (live 0.7.2 server, 2026-10-17 21:22-22:30 UTC; `tools/launch_bridge_072.py --bridge-port 5556 --game-port 25572`, then the motion, geometry, reset, crafting and combat differentials named in the report); this entry points at its table (a) rows 11
- server: Hytale 0.7.2 (HytaleServer.jar revision d769dc3d, SHA-256 23f71f03...0aff); bridge jar sha256 686BC4BC3DAE481AE857875961DAE75C02E2F4102B75B9B4193E60246EF80012
- receipt: `HytaleRL/hytale-plugin-0.7.2/FIDELITY-0.7.2.md` (commit cdeace2), table (a) rows 11; raw receipts in that report's `R/` folder (outside the repository)
- result: live 0.7.2 catalog: 2,047 recipes (same semantic hash on seeds 570057 and 1194077737), 22 distinct bench ids, campfire 3 recipes at 2.0 s; the simulator's table holds 1,947 and its packer rejects the 0.7.2 catalog (988 excluded-material inputs, 19 recipes without a bench); the shipped crafting compare refuses on 0.7.2 sources
- verdict: fail: the catalogue does not match

## 2026-10-17 — Source-pin audits: fail
- command: the gen-7 fidelity check by another session (live 0.7.2 server, 2026-10-17 21:22-22:30 UTC; `tools/launch_bridge_072.py --bridge-port 5556 --game-port 25572`, then the motion, geometry, reset, crafting and combat differentials named in the report); this entry points at its table (a) rows 12 and table (b)
- server: Hytale 0.7.2 (HytaleServer.jar revision d769dc3d, SHA-256 23f71f03...0aff); bridge jar sha256 686BC4BC3DAE481AE857875961DAE75C02E2F4102B75B9B4193E60246EF80012
- receipt: `HytaleRL/hytale-plugin-0.7.2/FIDELITY-0.7.2.md` (commit cdeace2), table (a) rows 12 and table (b); raw receipts in that report's `R/` folder (outside the repository)
- result: 0/11 audits pass on 0.7.2 and 11/11 on the 0.5.10 control; four crash before a report on classes 0.7.2 removed (LocalCachedChunkAccessor, BlockAccessor, BlockHealthModule$BlockHealthSystem) or on role assets that moved to Faction/ and Instructions/Combat/
- verdict: fail: re-baselining (P5) needs code changes, not only new hashes

## 2026-10-17 — Contracts and checkpoints: pending
- command: the gen-7 fidelity check by another session (live 0.7.2 server, 2026-10-17 21:22-22:30 UTC; `tools/launch_bridge_072.py --bridge-port 5556 --game-port 25572`, then the motion, geometry, reset, crafting and combat differentials named in the report); this entry points at its table (a) rows 13
- server: Hytale 0.7.2 (HytaleServer.jar revision d769dc3d, SHA-256 23f71f03...0aff); bridge jar sha256 686BC4BC3DAE481AE857875961DAE75C02E2F4102B75B9B4193E60246EF80012
- receipt: `HytaleRL/hytale-plugin-0.7.2/FIDELITY-0.7.2.md` (commit cdeace2), table (a) rows 13; raw receipts in that report's `R/` folder (outside the repository)
- result: no gen-7 checkpoint exists; only rulesets/hytale_0_5_7/ exists, and server_versions/v0_7.py accepts only 0.5.7 checkpoints
- verdict: pending: nothing to transfer yet (P6)

## 2026-10-17 — Legal observation: pending
- command: the gen-7 fidelity check by another session (live 0.7.2 server, 2026-10-17 21:22-22:30 UTC; `tools/launch_bridge_072.py --bridge-port 5556 --game-port 25572`, then the motion, geometry, reset, crafting and combat differentials named in the report); this entry points at its table (a) rows 14
- server: Hytale 0.7.2 (HytaleServer.jar revision d769dc3d, SHA-256 23f71f03...0aff); bridge jar sha256 686BC4BC3DAE481AE857875961DAE75C02E2F4102B75B9B4193E60246EF80012
- receipt: `HytaleRL/hytale-plugin-0.7.2/FIDELITY-0.7.2.md` (commit cdeace2), table (a) rows 14; raw receipts in that report's `R/` folder (outside the repository)
- result: no native differential is defined; the simulator's legal mode exists (observation/v3 encoder) and is off by default
- verdict: pending: the three-world leakage test is skill_agent work, not yet built
