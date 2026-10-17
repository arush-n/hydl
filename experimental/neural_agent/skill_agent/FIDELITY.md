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
