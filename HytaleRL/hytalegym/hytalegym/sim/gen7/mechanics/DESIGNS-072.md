# Gen-7 construct designs, lane stubs_69 (offline build, 2026-10-18)

Placed by the coordinator on 2026-10-18 (commit after 0161a16). The probes named below live at
`HytaleRL/hytalegym/tests/fidelity/combat/gen7_probes/` (the gitignored fidelity tree, on disk only; `_gen7_probe_common.py`
and `_on_demand.py` are their shared harness). The bytecode dumps the lane read (92 entries) stayed in the lane's scratch
directory and are not part of the repository; every claim below names its class or asset so it can be re-read from the jar.
Run order for the live measurements, on a quiet machine, is section 14a; the coordinator's decisions on 14b are in
`mechanics_072.md` (candidate rows) and `GEN7-PLAN.md`.

Lane: stubs_69 of the 0.7.2 (gen-7) simulator port. Repo HEAD e4a0833 at the start (renders read HEAD). Nothing was run
against a bridge, no server was launched, no test was run, and nothing in the repository was edited. The probes were only
import-checked (`--help`). Labels: **[C]** confirmed by a file read or a command output named in the line; **[I]** inferred
(the line says what would confirm it). Paths: `HYTALEGYM` = `/Users/arush/hydl/HytaleRL/hytalegym/hytalegym/`, `JAX` = `HYTALEGYM/jax/`.

## Summary (priority order)

| # | Mechanic | Switch (gen-7 value) | Hook target at HEAD (`JAX/`) | State needed | Receipt needed | Timing-sensitive | Probe (`tests/fidelity/combat/gen7_probes/`) |
|---|---|---|---|---|---|---|---|
| 1 | Hit interrupt, stun, flinch | `hit_interrupt` (True) | `combat/mechanics/runtime/kernel.py:98` `resolve_damage_events` (overlay Hook `hit_interrupt_flinch`, stub) | per-entity timers for Eligible, Flinch, Stunned, Recent_Reaction A and B, Recovery_Pending, Enraged, Protected_Attack; attack-cancel input | native NPC trace and learner row on Skeleton_Fighter, 3 swings | yes | `hit_interrupt.py` |
| 2 | Dodge action head | `dodge_action_head` (left, right) | `combat/observation/v3/policy/layout.py:659` `split_locomotion_choice`; sizes at `layout.py:194-198` from `schema/contract.py:123` (DODGE_ACTION_COUNT = 4) (overlay Hook `dodge_action_head`, stub) | none (head remap) | native advertised dodge offsets per tick | set: no; invulnerability and launch timing: yes | `dodge.py` |
| 3 | Rune casting (and the mana pool) | `rune_casting` (True); `mana_pool` (True, active) | `combat/arsenal/profiles/hytale_0_5_7.py:654` `hytale_0_5_7_loadouts` (Hook `rune_casting`, stub); mana `combat/mechanics/factory.py:81` `default_mechanics_rules` (Hook `mana_pool`, active) | RESOURCE_MANA (existing), a rune slot, ability charge recharge (existing `ability_charge_times_seconds`) | mana series and ability rows per cast, on a rune loadout (none exists) | yes | `rune_casting.py` |
| 4 | NPC set-velocity knockback | `npc_velocity_model` (role_set_velocity) | none verified: the 0.7.2 path is `Role.processSetVelocityInstruction` (only caller: `NPCVelocityInstructionSystem`); whether an agent hit reaches it is open; sim analogue `kernel.py:536` `apply_damage_forces` (force-mode branch 553-556); no Hook | one `applied_velocity` per entity (existing); the 0.5.10 list needs a structural change | opponent displacement after agent swings and after NPC-on-NPC contact (`npc_velocity.py --agent-swings`) | yes | `npc_velocity.py` |
| 5 | Walk minimum descent rate | `walk_descent` (minimum_descent_rate) | premise unsupported: 0.7.2 `computeNewFallSpeed` has no constant descent floor; if a walk-down difference is measured, `combat/runtime/agent.py:845` (clip 901-905); no Hook | none (existing fall speed) | a slope or stairs fixture (none exists); the wrapper runs the ledge baseline only | yes | `walk_descent.py` (wrapper; baseline only) |
| 6 | Slope and wall limits | `slope_limits` (False, decided) | none: no 0.7.2 reader of the keys | none | jar scan (offline) | no | `slope_limits.py` |
| 7 | Block health reset on write | `block_health_reset_on_write` (True) | the 0.7.2 write path is `BlockOperations.setBlock` → `BlockHealthModule.onBlockReplaced` → `clearHealth`; the sim's block-write functions are not yet listed; `combat/block_interactions/adapters/world_binding.py:382-410, 492-494` (health reads); no Hook | `block_health_valid` per cell (existing) | surrogate mutable-block compare on 0.7.2 | no (event-driven) | `block_health.py` (wrapper) |
| 8 | Climbing (on demand) | `climbing` (True) | not a 0.7.2 change: the NPC walk controller reads the climb speed keys in both versions; the 0.7.2 player keys are read by no server class; reclassify as a gen-5 engine gap (section 8b); `world/entities/movement_states.py:344, 457` | climbable-cell lookup | ladder fixture | yes | `climbing.py` (offline facts; native blocked) |
| 9 | Doors (on demand) | `doors` (`world_v0_5`, kept) | `world/surrogate/_lookup.py:340` `apply_surrogate_door_use`; `world/surrogate/features/interactions.py:286` `apply_surrogate_door_intent` | door state (surrogate) | door compare after P5 re-baseline | no | `doors.py` (offline facts; native blocked) |
| 10 | Swimming (on demand) | `swimming` (`fluidfx_v0_5`, kept) | `world/entities/swimming.py` (NPC dive; FIDELITY row 10c cites line 112) | fluid medium | swimming compare after P5 | yes | `swimming.py` (offline facts; native blocked) |
| 11 | Gliding (on demand) | `gliding` (True) | `world/entities/physical_state.py:91-129` (gliding input excluded from the fall branch); no glide motion | glide state, GlidingActive | glider loadout and a fixture | yes | `gliding.py` (offline facts; native blocked) |
| 12 | Riding (on demand) | `riding` (True) | none (no mount state in `jax/`) | mount state | mount fixture | yes | `riding.py` (offline facts; native blocked) |
| 13 | Hookshot (on demand, P3) | `hookshot` (True) | none: no server class names it in either jar; `kernel.py:536` moves targets only | hook projectile, attach cell, pull (mechanism open) | hookshot or rune loadout | yes | `hookshot.py` (offline facts; native blocked) |

### Render status of these ten switches (confirmed)

Overlay line numbers below are from the working tree: `git status` shows `sim/gen7/overlay.py` and `sim/gen7/profile.py` as
modified by other lanes, so HEAD may differ. Engine (`JAX/`) line numbers are from `git show HEAD`.

- `common/render.py:384` inserts only hooks whose status is `active`. `gen7/overlay.py` has `hit_interrupt_flinch`, `rune_casting`
  and `dodge_action_head` at status `stub` (lines 689-692, 701-703, 791-793). The gen-7 render therefore runs **gen-5 behaviour**
  for hit interrupt, runes and the dodge head. [C]
- `npc_velocity_model`, `walk_descent` and `block_health_reset_on_write` have no Hook entry in `gen7/overlay.py`; `climbing`,
  `gliding`, `riding` and `hookshot` are set `True` in `gen7/profile.py` and have no Hook entry either. [C: grep of `Hook("`]
- Stub signatures do not match the PORTING step-7 contract (`construct(original, profile)`, returns `original` when the switch is
  off): `hit_interrupt.py` has `apply_on_hit`, `mana.py` has `cast`, `dodge.py` has `action_head`, `npc_velocity.py` has
  `apply_set_velocity`, `walk_descent.py` has `descent_velocity`, `block_health.py` has `on_block_write`, `slope_limits.py` has
  `limit`. None defines `construct`, so the overlay strings `hit_interrupt:construct`, `mana:construct` and `dodge:construct` would
  fail to resolve if their status were flipped to `active`. [C: read the stub files] Each stub needs the `construct` wrapper before
  its status changes (the `guard_cooldown.py` pattern).

---

## 1. Hit interrupt, stun and flinch (`Constructs.hit_interrupt`; phase 4, row 5a; priority 1)

**Stakes:** low-blast for the render (a single switch, gen-5 default), but it changes melee outcomes for 31 attack rows and the
opponent roles that carry eligibility.

### 1a. Authored facts (0.7.2)

- 31 interaction rows parent `DamageEntityParent_Hit_Interrupt` (Abilities 4: Fireball, GroundSlam x2, WindStrike; Battleaxe 5;
  Daggers 9; Mace 7; Sword 6). [C: `sim/gen7/data/weapons.json`, `parent_chain`, 31 rows; `R/02_static_mechanics_receipts.txt` section 2]
- The on-hit gate `Server/Item/Interactions/Weapons/DamageEntityParent_Hit_Interrupt_On_Hit.json` (asset read): a Serial runs
  `DamageEntityParent_On_Hit` and then, only if the target has `Hit_Interrupt_Eligible` and none of `Hit_Interrupt_Flinch`,
  `Hit_Interrupt_Enraged`, `Hit_Interrupt_Recovery_Pending`, runs a Serial of: `Hit_Interrupt_Cancel_Attack`
  (`Type: Interrupt`, `Entity: Target`, `RequiredTag: Attack`), `Hit_Interrupt_Apply_Stun`, `ApplyEffect Hit_Interrupt_Flinch`
  on the target, `Hit_Interrupt_Record_Reaction`. [C: asset read, `assets_extract/072/...`]
- Effects (asset read; the comments are authored text): `Eligible` 1.0 s, refreshed every 0.25 s by the NPC instruction ("expires
  within one second after the component stops running"); `Flinch` 0.5 s, `HorizontalSpeedMultiplier` 0 ("Stop walking for 0.5
  seconds. Gravity and knockback remain active"); `Stunned` 0.7 s ("Block attack starts for 0.7 seconds from an accepted hit.
  This delay includes the flinch") (the design table said only "stun 0.7 s"); `Recent_Reaction_A` and `_B` 4.0 s each; `Recovery_Pending`
  2.0 s ("Queue protection after the third flinch"); `Enraged` 8 s with a particle; `Protected_Attack` 8.0 s ("Mark an attack that
  starts during protection"). [C: `Hit_Interrupt_*.json`]
- Reaction slots (asset read of `Hit_Interrupt_Record_Reaction.json`): the first accepted hit applies A, the second applies B, and the
  third (A and B both present) clears both and applies `Recovery_Pending`. [C]
- Enraged (asset read of `Component_Instruction_Hit_Interrupt.json`, instruction 3): when the NPC has `Recovery_Pending` and neither
  `Flinch` nor `Enraged`, it applies `Enraged`. So Enraged starts when the flinch of the third hit ends (about 0.5 s after the hit). [C]
- Protected attacks: `Begin_Protected_Attack` marks the User's attack when the User has `Enraged`; `Finish_Protected_Attack` clears
  `Enraged` and `Protected_Attack` when both are present. No row in `weapons.json` references either. The reader of
  `Protected_Attack` is not located: it is not in the on-hit gate. [C: no referencing row in the table; the gate read] [I: the
  NPC attack files outside the extracted subset reference it; confirm with `grep -rl Begin_Protected_Attack` over `Server/`]
- Eligibility: 22 role files reference `Hit_Interrupt`; Skeleton_Fighter and Goblin_Scrapper reach `Component_Instruction_Hit_Interrupt`;
  Trork_Brawler, Trork_Unarmed and Kweebec_Razorleaf do not. [C: `R/02_static_mechanics_receipts.txt` section 2, static reading by
  the fidelity lane; the role rows exist in `sim/gen7/data/npc_roles.json`]. The simulator's current opponents (Trork_Brawler,
  Trork_Unarmed, Kweebec_Razorleaf) are therefore not interruptible. [C by the same receipt; FIDELITY row 5a]
- Against 0.5.10: no file mentions `Hit_Interrupt` (`R/02_static_mechanics_receipts.txt`, section 2). [C]
- Java side [C: report dumps for Q1; lane checks noted]: no class in either jar names the effect ids, the NPC parts or the fragments
  `Hit_` and `_Interrupt` (zero hits in the constant-pool scans). The cancel is generic: `InterruptInteraction.firstRun` calls
  `InteractionManager.cancelChains`, which sets the chain to Failed, sends a cancel packet and recurses into forked chains; identical
  in both versions. The apply is generic: `ApplyEffectInteraction.firstRun` calls `EffectControllerComponent.addEffect`; the 0.7.2-only
  `durationAttribute` path does not apply, because the Hit_Interrupt effects have a plain `Duration` [C: lane check of the asset files].
  `EffectConditionInteraction` reads effects with Match All and None, identical in both. The Enraged count is data: NPC instruction 3
  applies Enraged [C: lane check of `Component_Instruction_Hit_Interrupt.json`]; no Java counter exists or is needed. The NPC parts
  are data-only [I].

### 1b. Construct design

- **Hook:** `combat/mechanics/runtime/kernel.py:98` `resolve_damage_events` (the overlay names this function for `hit_interrupt_flinch`).
  Wrap it: the original returns `(state, health, resolution)`; after it, for each target whose `DamageResolution` row has
  `applied_damage > 0`, `blocked` false and `invulnerable` false (fields at `mechanics/schema/types.py:193-213`), and whose target
  role is eligible, run the on-hit chain. [I: the acceptance test is the DamageEntityParent_On_Hit success; the 0.7.2 gate tests
  effect presence, not damage, so confirm on the first native run]
- **Chain, in order:** (1) cancel the target's current attack root that carries the Attack tag (Java: the generic `InterruptInteraction` path, identical in both versions): the engine's interrupt input is
  the `ability_interrupted_by_type_mask` family (`arsenal/schema/types.py:134-137`), which only reads the actor's own inputs
  (FIDELITY row 5a); the construct needs an external interrupt flag for the target. [C: the mask's scope; I: the flag's form]
  (2) stun: block the target's attack starts for 0.7 s. (3) flinch: target horizontal speed x0 for 0.5 s; the engine already has a
  status speed channel (`arsenal/effects/areas.py:321-323`, `status_speed_multiplier`), which is the natural carrier. [C: channel
  exists; I: it applies to NPC targets]. (4) reaction: A, then B, then clear both and set `Recovery_Pending`. (5) Enraged 8 s when
  `Recovery_Pending` is set and no flinch is active (the NPC instruction's rule). (6) Protected_Attack: mark attacks started while
  Enraged; the reader is open (see 1a).
- **State:** per-entity float32 countdowns beside the mechanics state: eligible (1.0 s), flinch (0.5), stunned (0.7), recent_a and
  recent_b (4.0), recovery (2.0), enraged (8.0), protected (8.0), and the eligibility refresh timer (0.25 s). Add them the way the
  guard countdown was added: a Patch on `combat/mechanics/schema/types.py` and `factory.py` (`overlay.py:437-457` is the pattern).
  Eligibility is a per-role boolean from the gen-7 table (`npc_roles.json` `component_references` reaching
  `Component_Instruction_Hit_Interrupt`). [C: pattern exists; I: exact field names]
- **Gen-5 path:** `construct(original, profile)` returns `original` when `profile.constructs.hit_interrupt` is False. [C: PORTING
  step 7; `guard_cooldown.py` pattern]. The stub must be renamed to that signature (see the render-status note).
- **Behind which switch:** `Constructs.hit_interrupt` (`common/profile.py`). Flip the overlay Hook `hit_interrupt_flinch` from
  `stub` to `active` under the coordinator's lock (`sim/gen7/.overlay.lock`). [C: PORTING, "Several construct builders at once"]
- **Tests that pin it:** `tests/sim/test_hit_interrupt.py` (new, per lane): the chain on a synthetic `DamageResolution` for an
  eligible and an ineligible target; the reaction slot order; Enraged after the third flinch; gen 5 returns the original. Add the
  switch to `IMPLEMENTED_SWITCHES` in `tests/sim/test_constructs.py`. Gate: `tests/sim/test_render.py tests/sim/test_constructs.py`.

### 1c. Native receipt and probe

- **Probe:** `tests/fidelity/combat/gen7_probes/hit_interrupt.py` (intended repo path `HytaleRL/hytalegym/tests/fidelity/combat/gen7_probes/hit_interrupt.py`).
  Uses the lifecycle harness's `NativeActorLifecycleTrace.start` (exactly one NPC of the role in the roster), the learner-row
  `run_program` with ability slot 0 (`ability_none_plus_slots=1`) every 14 ticks, 3 requests, 300 ticks, seeds 570057 and 913771.
- **One command (quiet machine, bridge pair launched by the coordinator):**
  `/Users/arush/hydl/.venv/bin/python /Users/arush/.claude/jobs/31947bd7/tmp/stubs_69/probes/hit_interrupt.py --bridge-port <P> --game-port <G> --target-role Skeleton_Fighter --swings 3 --swing-period 14 --steps 300`
- **Receipt contents:** per tick the learner legal masks, actor state, abilities; the target's NPC trace (`lifecycle_events` with
  `kind`, `key`, `value_before`, `value_after`; `damage_events`; `state`; `attack_action_active`); a `summary` list of every
  `Hit_Interrupt_*` status event with its trace index and Java tick. The status key is the effect asset id (`StatusValue.id` from
  `asset.getId()`, `NativeNpcLifecycleCapture.java:529-531`). [C]
- **Expected 0.7.2 values (ticks at 1/30 s):** Flinch 15, Stunned 21, Recent_Reaction 120, Recovery_Pending 60, Enraged 240, Eligible
  refresh 30 (`EXPECTED_TICKS` in the probe). Expected order on the third hit: Flinch, Stunned and Recovery_Pending applied on one
  tick; Enraged added when the flinch ends. [C: assets; I: the same-tick order follows the Serial's order]
- **Blockers:** none known. That the bridge spawns Skeleton_Fighter as the opponent is unverified (the environment accepts any
  `npc_role`; the first run confirms it). The probe does not check that the
  iron sword's slot-0 chain lands on the target: the first run must show a `damage_events` row on the target.
  The `ability_none_plus_slots=1` and 14-tick spacing are [I] from the timing lane's iron sword receipts (claim 1, finish 14).

### 1d. Timing

- **Timing-sensitive:** all durations (they are seconds in the asset, counted in ticks), the cancel tick, the stun window, Enraged expiry.
  A loaded machine lengthens native ticks and shortens tick counts (FIDELITY (af)). Run on a quiet machine; record the header's load.
- **Snapshot facts (measurable under load):** the 31-row parent list, the gate chain, effect durations, eligibility by role.
- **Claim most likely wrong:** that the cancel is the only attack effect (the stun also blocks new starts). It is the asset's own
  comment, so it is confirmed; the open part is whether the cancelled attack restarts after Stunned.

---

## 2. Dodge action head (`Constructs.dodge_action_head`; row 9; priority 2)

**Stakes:** medium-blast. It changes the policy head width, which changes every checkpoint and every mask consumer (the owner's
rule retrains gen 7 from scratch, so no checkpoint is carried over). It does not change engine motion.

### 2a. Authored facts

- The dodge assets are identical in 0.5.10 and 0.7.2 (seven files; `R/02_static_mechanics_receipts.txt` section 1). [C]
- `Dodge.json` (0.7.2): MovementCondition branches `Left` -> `Dodge_Left` and `Right` -> `Dodge_Right`; `ForwardLeft`,
  `ForwardRight`, `BackLeft` and `BackRight` are `Simple` (no chain). [C: receipt section 1]
- `Dodge_Left`: Costs Stamina 2, force 13 `Set`, direction (-1, 0, 0), invulnerability 0.25 s; the regen delay effect Set -0.7 (FIDELITY
  row 9). [C: receipt section 1; FIDELITY row 9]
- So the **engine already executes only left and right** (`DODGE_AUTHORED_ACTION_MASK = (False, False, True, True)`,
  `combat/mechanics/schema/contract.py:70-81`; FIDELITY row 9). The gen-7 change is the policy head, not behaviour. [C]

### 2b. Construct design

- **Head layout at HEAD (confirmed):** the locomotion categorical `locomotion_gait_compass` has 37 entries: 1 idle, 4 gaits x 8
  compass directions, then the dodge entries from `ARSENAL_POLICY_LOCOMOTION_DODGE_START` (33). Dodge offsets 0, 1, 2, 3 are
  forward, back, left, right (`split_locomotion_choice`, `layout.py:659-673`; `ARSENAL_POLICY_LOCOMOTION_DODGE_START` at `layout.py:194`;
  size at `layout.py:197-198`). [C: `tests/fidelity/combat/gen7_probes/_gen7_probe_common.py` import check printed the head sizes and dodge start 33]
- **The trap (confirmed by the overlay's own note):** setting `DODGE_ACTION_COUNT` to 2 keeps the masked forward and back slots and
  drops left and right (`gen7/overlay.py` hook note; `observation/v3/schema/contract.py:123`). A change of width alone is wrong. [C]
- **Sites that read the width** (`DODGE_ACTION_COUNT` at HEAD): `observation/v3/encoding/differential.py:82`,
  `observation/v3/native/codec/decode.py:112` and `:169`, `observation/v3/policy/layout.py:326` and `:811` (encode clip),
  `observation/v3/schema/spec.py:165` and `:223`, and the 4-wide mask concat in `policy/surface.py:441-443` and
  `encoding/encoder.py:507` (from the overlay note; not re-read in this lane). [C for the grep list; I for the two not re-read]
- **Hook:** `combat/observation/v3/policy/layout.py:659` `split_locomotion_choice` (overlay Hook `dodge_action_head`), plus a Patch
  of the size constants at `layout.py:194-198`, so that gen-7 dodge offsets map to left (choice start + 0) and right (start + 1).
  [I: the exact remap; it must be checked against `native/codec/decode.py` before the flip]
- **State:** none. The head is a mapping.
- **Gen-5 path:** `construct(original, profile)` returns the original split when `dodge_action_head` is the gen-5 tuple
  `("left", "right", "forward", "back")`. [C: the profile default, `common/profile.py`]
- **Tests:** `tests/fidelity/combat/gen7_probes/dodge.py` receipt against the native advertised set; in-repo
  `tests/sim/test_dodge_head.py` (new): left and right round-trip through encode and decode; forward and back are not legal on gen 7;
  the gen-5 head is unchanged. Gate: `tests/sim/test_render.py tests/sim/test_constructs.py`.

### 2c. Native receipt and probe

- **Probe:** `tests/fidelity/combat/gen7_probes/dodge.py`. Four dodge requests (forward, back, left, right), 30 ticks apart, from a neutral start; records the
  legal dodge offsets on every tick (`advertised_dodge_ticks`), the requests, the actor position and yaw, and the invulnerability fields.
- **One command:** `/Users/arush/hydl/.venv/bin/python /Users/arush/.claude/jobs/31947bd7/tmp/stubs_69/probes/dodge.py --bridge-port <P> --game-port <G> --profile iron_sword --gap 30`
- **Expected:** advertised offsets 2 and 3 (left, right) on ticks where the dodge is legal; forward and back advertised or not is the
  open question the receipt answers. [I: the native legality of forward and back is not measured; the design assumes not advertised]
- **Blockers:** none beyond the bridge pair. The learner harness already has a left-dodge program (`DODGE_LEFT_PROGRAM` in `programs.py`).

### 2d. Timing

- Advertised set: a snapshot of the policy surface (not timing-sensitive). Invulnerability (0.25 s) and the launch tick: timing-sensitive
  (the overlay's Patch comments record a live 0.7.2 bridge measurement, jar 32EF5EDD: accepted tick 2, launch tick 3, stamina tick 4;
  the patches `dodge_starts_one_drain_later` and `dodge_effects_on_acceptance` carry it). [C: `gen7/overlay.py:381-398`, comment text]

---

## 3. Rune casting and the mana pool (`Constructs.rune_casting`; row 7b; `mana_pool` row 7a; priority 3)

**Stakes:** high-blast if built on the wrong path: a rune loadout changes the arsenal profile set and the learner head's slot
meaning. The mana pool is already active.

### 3a. Authored facts (0.7.2)

- 19 `Server/Item/Items/Rune/` rows in the gen-7 table: 8 ability rows (7 runes and `Rune_Empty`) and 11 modifier rows; the six
  `_Debug` runes under `Rune/Ability/_Debug` (2) and `Rune/Modifier/_Debug` (4) are not in the table. [C: `weapons.json` 19 `item:Rune*` rows;
  archive listing of `Server/Item/Items/Rune/` (25 entries including debug)]
- Ability rows (`ability` block, all `CostType: Mana`, `Slot: Core`): Rune_ChainHook Cost 50, Cooldown 6, Duration 0.8;
  Rune_ChargedShot 35 / 4; Rune_Enrage 45 / 4 (Duration 8); Rune_Fireball 30 / 2; Rune_GroundSlam 40 / 5; Rune_PoisonImbue 40 / 4
  (Duration 6); Rune_WindStrike 30 / 2 (Duration 8). [C: `weapons.json`, `item:Rune_*#ability`]
- Cast graph: `interaction:Ability_Fireball_Cast` is a `StatsCondition` with `UseAbilityCost` and Next and Failed branches; the root is
  `root:Root_Ability_Fireball`; the projectile is `projectile_config:Projectile_Config_Ability_Fireball` (`LaunchForce`, `Physics`).
  [C: `weapons.json` rows]. The fireball RunTime 0.87 s (mechanics_072.md) was not re-read here. [I]
- Mana: `Server/Entity/Stats/Mana.json` InitialValue 100, Min 0, Max 100, ResetType MaxValue, Additive +1 per 0.3 s while Alive;
  a second rule (Percentage 1.0 per 0.5 s with `CheckPlayerGameMode` Creative) also exists and is ignored for survival agents, as the
  mana pool docstring says. [C: `assets_extract/072/Server/Entity/Stats/Mana.json`; the Creative rule is common to every stat]. Gen 5:
  0 / 0, +1 per 0.2 s gated on NoDamageTaken 6 s and not Charging (`assets_extract/0510/.../Mana.json`). [C]

### 3b. Engine state at HEAD (what already exists)

- `RESOURCE_MANA = 1` (`combat/mechanics/schema/contract.py:15`). The gen-5 caster profiles already bind a `mana_cost` scalar to it:
  `_legacy_staff` (hytale_0_5_7.py:3645, binding at 3677), `_legacy_wand` (3724, 3758), `_legacy_spellbook` (3794, 3831), with
  `cost_kind=RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT`; the same cost helper serves `_gun_profile` (4755-4769, `mana_cost` and
  `mana_minimum`). [C: `git show HEAD` read] [I: the mana maximum defaults to 0 at HEAD (`factory.py:86`, per FIDELITY row 7a), so a
  staff cannot pay unless its profile sets one; confirm by reading the profile's `mana_maximum`]
- The 0.5.7 core profiles already include `flame_staff`, `ice_staff`, `stoneskin_wand`, `skeleton_mage_spellbook`, `root_wand`,
  `bombs`, `potions` (`hytale_0_5_7.py:96-127`). [C]
- The mana pool: `combat/mechanics/factory.py:81` `default_mechanics_rules`; maximum set from `mana_maximum` (line 126) and regen
  1.0 per 0.2 s at lines 133-134 (gen 5). The `mana_pool` construct is implemented (`sim/gen7/mechanics/mana_pool.py`, IMPLEMENTED
  True) and active in the overlay (Hook `mana_pool`, 7a). [C]
- Cooldowns: `ability_charge_times_seconds` in `arsenal/factory.py:124,387` and `arsenal/profiles/hytale_0_5_7.py:1073` [C]. A rune
  cooldown can use one charge that recharges over the rune's Cooldown. [I: that the recharge fits a single-charge slot; confirm in
  `arsenal/item_programs.py:77` before the design is final]

### 3c. Construct design

- **Mana pool (already active):** no change; the rune construct spends it.
- **Rune casting hook:** `combat/arsenal/profiles/hytale_0_5_7.py:654` `hytale_0_5_7_loadouts` (overlay Hook `rune_casting`, stub). The
  construct adds one Core-slot ability per rune row (the 7 ability runes), each with: a `mana_cost` scalar binding to `RESOURCE_MANA`
  (the existing legacy-caster pattern), a charge with the rune's Cooldown, and a launch of its projectile through the existing arsenal
  projectile runtime (`arsenal/projectiles/runtime.py`). [I: the projectile and effect path works for runes as it does for the staffs;
  the gen-5 staff pattern is the template]
- **State:** none new beyond the existing mana resource and the charge timers. The rune slot is a loadout row, not a state leaf.
- **Modifier runes:** 11 rows need `Server/Abilities` keyword semantics, which are not extracted (`PORTING.md`, hand-maintained table).
  Out of scope for the first construct: the seven ability runes only.
- **Gen-5 path:** the original loadouts (no rune rows). `construct(original, profile)` returns `original` when `rune_casting` is False.
- **Tests:** `tests/sim/test_runes.py` (new): cost deducted from `RESOURCE_MANA` at cast; cooldown blocks a second cast; the Fireball
  row values from `sim/gen7/data/weapons.json`; gen 5 returns the original. Plus `tests/sim/test_mana_pool.py` (existing).

### 3d. Native receipt and probe

- **Probe:** `tests/fidelity/combat/gen7_probes/rune_casting.py`. Requires `--profile <name>`, a learner loadout that holds a rune. **BLOCKED at HEAD:** no profile
  in `hytalegym/jax/combat/arsenal/profiles` carries a rune; the probe then records `cast_legal` never, which is the blocker, not a
  result. The native mana value is `info["native_agent_mana"]` (and `native_agent_mana_maximum`), published by
  `NativeEnvironmentSession.java:7506-7507`. [C]
- **One command (once a rune loadout exists):** `/Users/arush/hydl/.venv/bin/python /Users/arush/.claude/jobs/31947bd7/tmp/stubs_69/probes/rune_casting.py --bridge-port <P> --game-port <G> --profile <rune loadout> --ability-choice 1 --casts 3 --cast-period 60`
- **Expected for Fireball (per row):** mana drops by 30 at each cast (plus the refill of +1 per 0.3 s), the second cast is refused until
  2 s (60 ticks at 1/30 s), and the projectile follows. [C: row values; I: cast and flight timings, unmeasured]

### 3e. Timing

- Cost and cooldown are tick-counted on the native clock: a loaded machine changes tick numbers, not seconds. Refill is per 0.3 s of
  world time. **Timing-sensitive: yes.** Snapshot facts (rows, costs, cooldowns in the table) are measurable under load.
- **Claim most likely wrong:** that the staff's `mana_cost` binding can carry a rune cost unchanged. The staffs are gen-5 items whose
  mana maximum is 0 at HEAD; the first run of the probe on a rune loadout is the test.


---

## 4. NPC set-velocity knockback (`Constructs.npc_velocity_model`; no table row; P3)

**Stakes:** high-blast if built wrong (up to 23x knockback on one branch), and the sim has no confirmed path to it yet. Decide before building.

Provenance: **[C: lane check]** = I re-read the bytecode operands in the lane's dumps; **[C: report dump]** = taken from the bytecode report's dump files, not re-read; **[I]** = inferred.

### 4a. Authored facts (0.7.2 and 0.5.10 Java)

- The only caller of `Role.processSetVelocityInstruction` in either jar is `NPCVelocityInstructionSystem`. [C: report dump, `pool7.txt` and `pool5.txt` constant-pool scan]
- That system reads the `Velocity$Instruction` component, switches on the protocol `ChangeVelocityType` (Set = case 1, Add = case 2 in both jars), and calls the Set or the Add method. [C: lane check, `npcvis7_c.txt` lines 89-144; report dump for the case numbers]
- **Open:** what creates a `Velocity$Instruction` for a melee hit. No producer was traced. [open]
- 0.7.2 `processSetVelocityInstruction(Vector3d, VelocityConfig, float)` copies the vector, multiplies it by `knockbackScale` (JSON key `KnockbackScale`, default 1.0), and calls `setVelocity(v', cfg, false, pre)`. The float is not read. [C: lane check, `role7_c.txt` operations 12-19 and 33; report dump for the unread float]
- 0.5.10 `processSetVelocityInstruction(Vector3d, VelocityConfig)` calls `setVelocity(ZERO, null, false)` and then `addVelocity(vec, cfg)`. [C: lane check, `role5_c.txt`, full body]
- 0.5.10 `setVelocity(vec, cfg, boolean)`: if `SHOULD_MODIFY_VELOCITY` is false **and** cfg is non-null it clears `appliedVelocities` and appends one entry; otherwise it sets `externalVelocity = vec`. So the zero step with a null config only zeroes `externalVelocity` and leaves the list. [C: lane check, `mcb5_c.txt` 2929-2960]
- 0.5.10 `addVelocity(vec, cfg)` (`mcb5_c.txt` 2858-2928, operands read unfiltered):
  - default branch (flag false and cfg non-null): appends `AppliedVelocity(vec * ks, cfg)` on all three axes, without clearing. [C: lane check, offsets 0-58]
  - otherwise: `externalVelocity += (x*ks*k, y*ks, z*ks*k)` with `k = 0.18 * velocityResistance`. **The factor applies to x and z only; y has no k.** The report's "x, y and z" is wrong on this point. [C: lane check, offsets 59-122]
- 0.7.2 `setVelocity(vec, cfg, boolean, float)` (`mcb7_c.txt` 3339-3378, offsets 92-106 read): if the flag is false it clears `appliedVelocities`, zeroes `externalVelocity` and `appliedVelocity`, and appends one `AppliedVelocity(vec, cfg or playerDefaultMovementResistance())`. If the flag is true it sets `externalVelocity = vec` with no resistance factor. [C: lane check]
- `SplitVelocity` is a plugin with a static flag `SHOULD_MODIFY_VELOCITY`. Its **static initialiser sets it true**, its `setup()` sets it false, and `shutdown()` sets it true. So the live state depends on whether the plugin's `setup()` runs in the server. The bridge source does not reference the class. [C: lane check, `split7_c.txt` lines 5, 12-31; grep of the bridge source: no match] [I: a core plugin is set up at boot; confirm with the probe]
- Asset: `movement/config:Default` has `VelocityResistance` 0.242; so do `config:Goblin_Launcher_Immobile` and `config:Mount`. No NPC role row in `npc_roles.json` carries the key. Which MovementConfig an NPC body uses is **open**. [C: lane check, `sim/gen7/data/movement.json`; `npc_roles.json` has 0 rows with the key]

### 4b. What the numbers say, per branch

- **Default branch** (flag false, config non-null): the two versions give the same magnitude (`vec * ks` per axis). 0.7.2 replaces the queued applied velocities; 0.5.10 appends them. The difference shows only when two set instructions land before the list drains. [C: lane check of 4a]
- **Other branch** (flag true, or 0.5.10 with a null config): horizontal (x, z) knockback in 0.7.2 is `1 / (0.18 * vr)` times 0.5.10's. With `vr = 0.242`, that is 22.96, which is the 23x of TRAINING-0.7.2.md item 2. The vertical (y) component is equal in both. [C: arithmetic on the lane-checked operands and the asset value] So the 23x is reproduced **only** on this branch, and it comes from the asset value, not a code constant.
- The only measured displacement in the docs runs the other way: 0.19 m on 0.7.2 against 0.44 m on 0.5.10 (quoted in FIDELITY row 5b from TRAINING-0.7.2.md). That comparison also changed the sword knockback values (row 5b), so it does not isolate this path. [C: FIDELITY row 5b text]
- The live flag decides everything: default branch means no 23x. [open: item 1 in section 14]

### 4c. Construct design (not started; measure first)

- **Hook target, if the producer chain is confirmed:** `combat/mechanics/runtime/kernel.py:536` `apply_damage_forces` (the force-mode branch at 553-556: mode 0 replaces, mode 1 adds). That is the sim's knockback path. [I: the producer chain; without it the construct has no sim path and is inert]
- **State:** the sim keeps one `applied_velocity` vector per entity, with a per-event resistance (`applied_air_resistance` and the ground fields; `kernel.py` `apply_one`, lines 549-575). [C: lane check] The 0.5.10 behaviour is a list of decaying entries, each with its own config. Append semantics therefore need a list or an equivalent sum: a structural change, not a constant.
- **Gen-5 path:** `construct(original, profile)` returns the original `apply_damage_forces` when the switch is `zero_then_add`.
- **Tests, if built:** a two-event overlap (replace versus append) and the horizontal-only factor on the other branch (`tests/sim/test_npc_velocity.py`, new, per lane).
- **Recommendation:** do not build yet. The probe in 4d answers the two open questions (whether an agent hit reaches a set instruction; the live branch). Separately, the owner should review the `role_set_velocity` value and its "~23x" comment in `sim/gen7/profile.py`. This lane did not edit that file (lock protocol).

### 4d. Native receipt and probe

- **Probe:** `tests/fidelity/combat/gen7_probes/npc_velocity.py`. It traces every NPC in the bounds (one lifecycle trace each) and records damage events between two traced NPCs. With `--agent-swings N` the agent also swings at the opponent, so the opponent's displacement after an agent hit is in the receipt.
- **One command (quiet machine, bridge pair launched by the coordinator):**
  `/Users/arush/hydl/.venv/bin/python /Users/arush/.claude/jobs/31947bd7/tmp/stubs_69/probes/npc_velocity.py --bridge-port <P> --game-port <G> --npc-role Kweebec_Razorleaf --agent-swings 2 --swing-period 14 --steps 150`
- **Predictions to test [I]:** with two swings inside the decay window, the default branch gives replace semantics (the later displacement follows the later hit alone); the other branch gives horizontal displacement about 23x the 0.5.10 value. If neither shows, the agent hit does not reach a set instruction.
- **Blockers:** no NPC-on-NPC scenario exists at HEAD (the probe is observer-only for that case); the producer chain is open.

### 4e. Timing

- Displacement per tick: **timing-sensitive** (native tick length under load shifts it). Run on a quiet machine.
- Snapshot facts (the branch structure, the 0.242 asset value, the 0.18 constant) are measurable under load.

---

## 5. Walk minimum descent rate (`Constructs.walk_descent`; P2)

**Stakes:** medium-blast if built (NPC walk physics). The premise has no code behind it.

### 5a. Authored facts

- `MotionControllerWalk.computeNewFallSpeed(double, double)` in 0.7.2 [C: lane check, `mcw7_c.txt` 4031-4112]: the only numeric constants are 1, -1 (from `invertedGravity`), 0.001225 (the air-density term) and zero comparisons. Terminal velocity `T` is the gravity sign times `PhysicsMath.getTerminalVelocity(...)`. The speed is driven by `getAcceleration(v, T) * dt` and clamped between `v` and `T`. **There is no constant descent floor.** [C: lane check]
- So TRAINING-0.7.2.md item 8 ("0.7.2 forces a minimum descent rate when descending") is not supported by this method. [C: lane check]
- The same family differs elsewhere. [C: report dumps `pm5.txt`, `pm7.txt`, `cm7.txt`; not re-read]
  - `PhysicsMath.getAcceleration`: 0.5.10 is `32 * (1 - |v/T|^3)`; 0.7.2 is `32 * (sign(T) - sign(v) * |v/T|^3)`. Identical when falling; different when rising (a jump effect). [I: the physical effect]
  - `computeMove` in 0.7.2 also caps the vertical speed with `Math.min(vy, -fallSpeed * dt)`. The 0.5.10 counterpart is not checked. [open]
- **Overlap:** `npc_jump_arc` (`physics_math`) already models `computeNewFallSpeed` (mechanics_072.md row 42 cites this method). Any descent difference in that function sits inside that construct's scope. [C: design text] [I: that its descent branch is covered; check its tests]

### 5b. Construct design

- **Recommendation:** remove `minimum_descent_rate` from the walk_descent plan and keep `free_fall` unless a slope or stairs measurement shows a difference. The owner decides. [decision]
- **If a walk-down difference is measured:** hook `combat/runtime/agent.py:845` `_tick_agent_geometry_motion` (fall clip at 901-905: `jnp.clip(walk_fall_speed, 0.0, maximum_fall_speed)`), the gen-5 walk fall path. [C: lane check of the lines]
- **Gen-5 path:** returns the original.

### 5c. Native receipt and probe

- **Probe:** `tests/fidelity/combat/gen7_probes/walk_descent.py`, a wrapper over the adk motion tool. It runs only the existing flat and 3-block ledge baseline. No slope or stairs fixture exists, and a ledge walk-off is a fall, so the probe **cannot test the premise**. [C: the tool's arguments; FIDELITY row 1e]
- **Blocker:** a slope or stairs fixture, then a walk-down comparison.

### 5d. Timing

- The per-step bar is 5e-4 m, and long native ticks break it (FIDELITY row 1c): **timing-sensitive**. The bytecode is a snapshot.

---

## 6. Slope and wall limits (`Constructs.slope_limits` = False; decided 2026-10-18)

- `MaxSlopeAngleDegrees` and `MaxWallAngleDegrees`: two class hits each in 0.7.2 (`MovementConfig` and the protocol `MovementSettings`), zero in 0.5.10. [C: report dumps `pool7.txt`, `pool5.txt`; not re-read]
- `MovementConfig$1` writes the two lowercase fields and reads neither. No class references the getter or the field name. [C: report dump `pool7b.txt`] [I: server code ignores the keys]
- **Decision holds:** no reader, so `slope_limits` stays False. `tests/fidelity/combat/gen7_probes/slope_limits.py` repeats the byte search over both jars and needs no bridge. It was not run in this lane (the bytecode report already did the same search).
- Timing: a snapshot, not timing-sensitive.

---

## 7. Block health reset on write (`Constructs.block_health_reset_on_write`; P5)

- **0.7.2 write path:** `BlockOperations.setBlock` calls `BlockHealthModule.onBlockReplaced` (`bops7_c.txt` offset 312; `bhm7_c.txt` line 100). That calls `BlockHealthSection.clearHealth(int)` (`bhm7_c.txt` line 118), which removes the health and pending entries of that index (`bhs7_c.txt` line 206). [C: lane check of the call chain] [C: report dumps for the method bodies]
- **Regeneration:** an entry is removed when its health reaches 1.0 (the `fconst_1` compare in `tickRegeneration`, `bhs7_c.txt`). [C: report dump; not re-read]
- **Wording fix:** the mechanics doc says the entry is dropped "once health leaves (0, 1)". The bytecode says: removed when regeneration reaches 1.0, and on any `setBlock` write. [C: report dumps against the design text]
- **Placement:** `PlaceBlockEventSystem` sets fragility from `WorldConfig.getBlockPlacementFragilityTimer()` in both versions. [C: report dumps `bhpb7_c.txt`, `bhpb5_c.txt`]
- **0.5.10:** `BlockHealthChunk.removeBlock(World, Vector3i)` removes the entry and notifies the world. [C: report dump `bhc5_c.txt`] `BlockHealthModule$BlockHealthSystem` exists only in 0.5.10: the 0.7.2 class listing has no such class, while `BlockHealthSystems` and its nested classes exist in 0.7.2. [C: lane check of the class listings] The 0.5.10 callers are constant-pool references only, so the call paths are [I].
- **State:** the sim already has `block_health` and `block_health_valid` per cell (`combat/block_interactions/adapters/world_binding.py` 382-410 and 492-494). [C: lane check]
- **Design:** every block write in the sim clears validity, not only harvest; and validity is cleared when regeneration brings health to 1.0. **Open:** the sim's list of block-write functions (map every setter before the hook is placed).
- **Gen-5 path:** returns the original.
- **Native:** `tests/fidelity/combat/gen7_probes/block_health.py` wraps `tests/fidelity/surrogate_mutable_block_native_compare.py`, which walks initial, damaged, repaired and removed states. The transitions are event-driven, not tick-counted.

---

## 8. Climbing (`Constructs.climbing`; on demand)

**Stakes:** medium-blast. **Classification changed by the bytecode:** this is not a 0.7.2 change for the NPC body (see 8a).

### 8a. Authored facts

- The NPC walk controller reads `ClimbSpeed`, `ClimbSpeedMult`, `ClimbSpeedPow` and `ClimbSpeedConst` in **both** versions: `MotionControllerWalk` in 0.5.10 and 0.7.2 both carry all four names, and 0.7.2 calls `computeClimbSpeed(D)` (`mcw7_c.txt` line 705). [C: lane check of name counts in `mcw5_c.txt` and `mcw7_c.txt`] The builder and fly controllers read the same keys. [C: report dump `pool7.txt`, `pool5.txt`]
- `ClimbSpeedLateral`, `ClimbUpSprintSpeed` and `ClimbDownSprintSpeed` (the 0.7.2 player keys in `movement/config:Default`) appear only in `MovementConfig` and the protocol `MovementSettings`. **No server class reads them.** [C: lane check, `pool7.txt` lines 29 and 32 name them; report dump for the reader list]
- `IsClimbable` appears only in the block-movement classes. [C: report dump] 57 block rows mention `climb` in `sim/gen7/data/blocks.json`. [C: lane check]
- Engine (HEAD): `jax/world/entities/movement_states.py` resets the climbing bit at line 344 and sets it at line 457 only for `ascent_climb`. The NPC-animation reading of that bit is [I]. [C: lane check]
- The engine has no climb move for the agent (mechanics_072.md, phase 3 status). [C: design text]

### 8b. Classification and design

- **Consequence:** the NPC climb speed path exists in both versions, so the sim's missing climb move for the bridge body is a **gen-5 engine gap** that affects both generations. It is not a gen-7 construct. [I: the bridge body uses the NPC walk controller, FIDELITY row 1a and mechanics_072 phase-3 status]
- **Recommendation:** reclassify climbing as an engine gap for every generation (the rule mechanics_072 already states for fixes shared by all generations, F1c). Keep `climbing` out of the gen-7 switch list unless the owner wants a placeholder. [decision for owner]
- If built, the hook is the agent's vertical motion beside `combat/runtime/agent.py` (the walk fall path at line 845) plus a climb state in `world/entities/movement_states.py`. [I]

### 8c. Native receipt and probe

- **Probe:** `tests/fidelity/combat/gen7_probes/climbing.py`. The offline facts run now (exit 0). `--native` exits 3: **blocked** (no climb choice in the learner v3 head; no ladder fixture; no climb-capable actor test).
- **Receipt once unblocked:** per tick, position and movement state on a ladder, for each speed.

### 8d. Timing

- Climb speed is per tick: **timing-sensitive**. The key lists and the block count are snapshot facts.

---

## 9. Doors (`Constructs.doors` = `world_v0_5`, kept; on demand, P5 then P2)

**Stakes:** low-blast for now (the gen-5 model is kept).

### 9a. Authored facts

- 108 rows of `blocks.json` contain the substring `Door`. A substring count: it includes non-door names, so read the sample before using it. [C: lane check, offline receipt `doors_static_facts.json`]
- The door source audit crashes on 0.7.2 with `KeyError` for `BlockAccessor.class` (FIDELITY row 10b, cited there from `R/live/03j_door_compare.out`). [C: FIDELITY row 10b text]
- `DoorBlockUtils` exists only in 0.7.2. It holds the `OPEN`/`CLOSE` `DOOR_IN`/`OUT` constants and an enum `DoorState {CLOSED, OPENED_IN, OPENED_OUT}`. In 0.5.10 the same three states are nested in `DoorInteraction$DoorState`. [C: report dump, in-session output] The state model is the same; the move reads as a refactor. [I]

### 9b. Engine (HEAD) and design

- Surrogate door use and intent: `jax/world/surrogate/_lookup.py:340` (`apply_surrogate_door_use`) and `jax/world/surrogate/features/interactions.py:286` (`apply_surrogate_door_intent`). [C: lane check of the grep at HEAD]
- The gen-7 profile keeps `doors="world_v0_5"`. [C: `gen7/profile.py`]
- **Design:** none until P5 re-baselines the door audit and a differential differs (the decision in `mechanics_072.md`).

### 9c. Native receipt and probe

- **Probe:** `tests/fidelity/combat/gen7_probes/doors.py` (offline facts now; `--native` exits 3). **Blocked:** `surrogate_door_native_compare.py` refuses unless the door audit matches 0.5.7 (FIDELITY row 10b).
- **Receipt once unblocked:** door transitions for an open and a close request, from the door compare.

### 9d. Timing

- Door state changes are event-driven: not timing-sensitive in principle.

---

## 10. Swimming (`Constructs.swimming` = `fluidfx_v0_5`, kept; on demand, after P5)

**Stakes:** low-blast for now (the gen-5 model is kept).

### 10a. Authored facts

- No `Swim*` class exists in either jar. [C: lane check of the class listings (my earlier name filter matched `Swim` and returned none in 0.7.2); report dumps for 0.5.10]
- `SwimJumpForce` is in both `MovementConfig` classes. [C: report dump `mc7_c.txt`]
- 0.7.2 adds `GroundedConditionInteraction`, which is not a swim class. [C: report]
- So any swim difference is in data (the MovementConfig values), not in code. The FIDELITY row 10c catalogue values (swim x0.6, up and down 2.5, sinking 1.35) were not re-read. [C: cited text]
- The swimming compare asserts server 0.5.7 (FIDELITY row 10c). [C: cited text]

### 10b. Engine and design

- Only the NPC dive is modelled (`jax/world/entities/swimming.py`, per FIDELITY row 10c). Observation flags `swimming` and `swim_jumping` exist (`observation/v3/schema/contract.py` 101-102). [C: lane check of the grep at HEAD]
- **Design:** none until the swimming compare is re-baselined (P5), and the MovementConfig values are compared.

### 10c. Native receipt and probe

- **Probe:** `tests/fidelity/combat/gen7_probes/swimming.py` (offline facts now; `--native` exits 3). **Blocked:** `surrogate_swimming_native_compare.py` asserts `native_server_version == "0.5.7"`.

### 10d. Timing

- Swim speed is per tick: **timing-sensitive** once a compare runs.

---

## 11. Gliding (`Constructs.gliding`; on demand)

**Stakes:** medium-blast (a movement state and a stamina rule).

### 11a. Authored facts

- `item:Glider_Scrap` is in `weapons.json`. [C: lane check, offline receipt]
- `stat:GlidingActive` (Max 1; `+1` and `-1` per 0.1 s under the Gliding condition). [C: receipt `R/02_static_mechanics_receipts.txt` section 3; lane check of `stats.json`]
- `Stamina` (0.7.2): `+0.3` per 0.1 s under Stat, Stat, Wielding, Sprinting and Gliding; `-0.1` per 0.1 s under Gliding. `StaminaRegenDelay`: `+0.1` per 0.1 s under Gliding, Gliding. Stamina differs from 0.5.10; StaminaRegenDelay is identical. [C: receipt section 3]
- Java: `GlidingCondition` reads `MovementStates.gliding`. A 0.7.2 `ConditionInteraction` adds a gliding key absent in 0.5.10. `ItemGlider` adds `EnergyClimbRate` and `DiveSpeedToEnergyRatio`, and no other class reads them. `GlidingActive` has zero code hits. [C: report dumps `pool7b.txt`, `pool5b.txt`, `pool7e.txt`, `pool5e.txt`, `tgi7_c.txt`]
- **Open:** the stamina drain reader was not found. [open]
- Glide speed 10, falls at most 2 per second, stamina 1 per second (catalogue, FIDELITY row 10d; not re-read). [C: cited text]

### 11b. Engine (HEAD) and design

- `jax/world/entities/physical_state.py` lines 91-129 take a `gliding` input and keep it out of the fall branch (`excluded` at line 129). There is no glide motion. [C: lane check of the lines]
- **Finding:** no server class reads the glider energy fields or a glide speed that I could find, so the glide speed and fall rate may be client-side or in a class not traced. [I]
- **Design:** a glide state (speed, fall rate, stamina rule) behind `Constructs.gliding`, once the native measurement shows what the server does.

### 11c. Native receipt and probe

- **Probe:** `tests/fidelity/combat/gen7_probes/gliding.py` (offline facts now; `--native` exits 3). **Blocked:** no learner profile carries `Glider_Scrap`, and the learner v3 surface has no gliding choice.

### 11d. Timing

- Glide speed and fall rate are per tick: **timing-sensitive**. The stamina rules are snapshot facts.

---

## 12. Riding (`Constructs.riding`; on demand)

**Stakes:** medium-blast (an actor-mount state).

### 12a. Authored facts

- `movement/config:Mount` (`MovementConfig/Mount.json`, new in 0.7.2) has a `fields` block with the slope and wall keys (30 and 15) and `ClimbSpeed` 0.035. [C: lane check, `movement.json`] The rider catalogue numbers (speed 8, sprint 13.2, jump 14; FIDELITY row 10e) were not re-read. [C: cited text]
- Java: `MountNPC` and `MountMovement` packets; `NPCMountSystems$OnAdd` references `MountNPC`. [C: report dump `pool7.txt`] No mount keys in the `MovementConfig` class. [C: report, in-session]
- `builtin/mounts` has 41 entries in 0.7.2 and 39 in 0.5.10. [C: report dumps `classes7.txt`, `classes5.txt`] The first copy of the report named the two additions as `MountSystems$MountedEntitySpectating` and `NPCMountSystems$DismountOnPlayerSpectating`. Those names read as spectating, not mount physics. [I: from the names only]

### 12b. Engine (HEAD) and design

- No mount state in `jax/` (grep at HEAD). [C: lane check]
- **Design:** none until a mount fixture exists and the 0.7.2 rider numbers are measured.

### 12c. Native receipt and probe

- **Probe:** `tests/fidelity/combat/gen7_probes/riding.py` (offline facts now; `--native` exits 3). **Blocked:** no mount fixture and no mount choice in the learner v3 surface.

### 12d. Timing

- Rider speed and jump are per tick: **timing-sensitive**.

---

## 13. Hookshot (`Constructs.hookshot`; P3, on demand)

**Stakes:** medium-blast (a projectile that pulls the actor; no engine path exists).

### 13a. Authored facts

- `item:Tool_Hookshot`, `projectile_config:Projectile_Config_Hookshot`, `interaction:Ability_ChainHook_Cast` and `_Hit`, `projectile_config:Projectile_Config_ChainHook`, `root:Root_Ability_ChainHook`. [C: lane check, `weapons.json`]
- `item:Rune_ChainHook`: cost 50 mana, cooldown 6, Duration 0.8, Core slot. [C: lane check, `weapons.json`]
- **Java: zero hits** for `Hookshot`, `ChainHook` or `Grapple` in either jar. [C: report dumps `pool7.txt`, `pool7b.txt`] So no server class names the hookshot. Its behaviour is either composed from generic interaction types in the asset data or absent from the server. [I]

### 13b. Engine (HEAD) and design

- No pull state or hook projectile in `jax/`. The knockback path (`combat/mechanics/runtime/kernel.py:536` `apply_damage_forces`) applies velocity to the **target** (`target_entity_id`), not the actor. [C: lane check]
- **Open:** the pull mechanism. The candidates are the generic knockback or force interaction types, if the asset composes the pull from them. [open]
- The chain hook is a rune, so it depends on the rune_casting work (section 3).

### 13c. Native receipt and probe

- **Probe:** `tests/fidelity/combat/gen7_probes/hookshot.py` (offline facts now; `--native` exits 3). **Blocked:** no hookshot loadout and no rune loadout at HEAD.

### 13d. Timing

- Fire, attach and pull timing are per tick: **timing-sensitive**. The asset rows are snapshot facts.

---

## 14. Run order, owner decisions, open items

Nothing in this file was run against a bridge. Every probe is import-checked (`--help`); the six verb probes and `slope_limits.py` also ran their offline parts into a lane scratch folder.

### 14a. Run order (quiet machine; the coordinator launches each pair with `tools/launch_bridge_072.py`)

1. `tests/fidelity/combat/gen7_probes/hit_interrupt.py --target-role Skeleton_Fighter --swings 3 --swing-period 14 --steps 300` (section 1c).
2. `tests/fidelity/combat/gen7_probes/dodge.py --profile iron_sword --gap 30` (section 2c).
3. `tests/fidelity/combat/gen7_probes/npc_velocity.py --npc-role Kweebec_Razorleaf --agent-swings 2 --swing-period 14 --steps 150`, then the same with `--agent-swings 0` as the observer-only control (section 4d).
4. `tests/fidelity/combat/gen7_probes/block_health.py` (section 7; wraps the existing compare).
5. `tests/fidelity/combat/gen7_probes/walk_descent.py --steps 30` (baseline only; section 5c).
6. `tests/fidelity/combat/gen7_probes/rune_casting.py --profile <rune loadout>`, only after a rune loadout exists (section 3d).
7. Offline now, no bridge: `tests/fidelity/combat/gen7_probes/slope_limits.py` (section 6), and the six verb probes without `--native` (sections 8-13).

### 14b. Owner decisions (not made by this lane)

- a. `npc_velocity_model`: the profile value `role_set_velocity` and its "~23x" comment in `sim/gen7/profile.py` (and the `mechanics_072.md` row). The 23x holds only on the non-default `SplitVelocity` branch with `velocityResistance` 0.242; on the default branch the magnitudes match (section 4b).
- b. `walk_descent`: the `minimum_descent_rate` premise has no code behind it (section 5a). Recommend keeping `free_fall`, or dropping the construct, until a slope measurement says otherwise.
- c. `climbing`: reclassify as a gen-5 engine gap for every generation (section 8b).
- d. `climbing`, `gliding`, `riding` and `hookshot` are set True in `gen7/profile.py` but have no Hook entry, so the render does not apply them. Keep them as placeholders or reclassify.
- e. Stub API: no stub defines `construct`, so flipping a Hook to `active` would fail at render. The three stub hooks (`hit_interrupt_flinch`, `rune_casting`, `dodge_action_head`) are `stub`, so the gen-7 render currently runs gen-5 behaviour for them (render status, section 0).
- f. `mechanics_072.md` wording for block health ("once health leaves (0, 1)") should read: removed when regeneration reaches 1.0, and on any `setBlock` write (section 7).

### 14c. Open items, each with a named check

1. Live `SplitVelocity` state. The static initialiser sets the flag true and `setup()` sets it false. Check: find the server's callers of the plugin lifecycle, or observe through `npc_velocity.py` (section 4b). This decides the 23x.
2. The producer of `Velocity$Instruction` for a melee hit (section 4a). Check: a constant-pool scan for the component's constructors and writers in both jars.
3. Which MovementConfig an NPC body uses. `VelocityResistance` 0.242 is on `config:Default`, but no NPC row carries the key (section 4a).
4. The 0.5.10 counterpart of the `computeMove` vertical cap `Math.min(vy, -fallSpeed * dt)` (section 5a).
5. The gliding stamina drain reader (section 11a).
6. The hookshot pull mechanism (section 13b). Check: which generic interaction types the Tool_Hookshot asset composes.
7. The sim's list of block-write functions (section 7).
8. The `Protected_Attack` reader (section 1a). Check: grep `Begin_Protected_Attack` over the full `Server/` tree of the install, not only the extracted subset.

### 14d. Claims most likely wrong

1. That the bridge's NPCs run on the non-default `SplitVelocity` branch. The 23x depends on it (open item 1). The alternative, that `setup()` runs and the default branch applies, would make the probe's 23x prediction fail and the 4b table wrong.
2. That the iron-sword chain at 14-tick spacing lands hits on the opponent. The spacing comes from the timing lane's receipts and was not run here. If the swings miss, the hit-interrupt and npc-velocity receipts show no events, and the 15, 21, 120, 60 and 240 tick expectations are untested.
