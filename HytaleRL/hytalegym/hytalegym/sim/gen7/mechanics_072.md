# Gen-7 constructs: what 0.7.2 does that the gen-5 engine does not (design, 2026-10-17)

A construct is one behaviour, implemented once in the engine behind one field of
`hytalegym.sim.common.profile.Constructs`. Gen 5 keeps every default (today's behaviour); gen 7
sets the field (`sim/gen7/profile.py`). The stubs in `mechanics/` are typed, unwired interfaces;
`mechanics.require_construct(profile, field)` raises `NotImplementedError` until a construct is
implemented, so a gen-7 render cannot quietly run with gen-5 behaviour. Under the owner's render
rule the gen-5 source is never edited: a render copies it from `git show HEAD:<path>`, applies the
redirect list (`sim/common/content/redirects.json`), and inserts a hook where a construct runs.

Gaps and measurements come from `HytaleRL/hytale-plugin-0.7.2/FIDELITY-0.7.2.md` (rows cited as
F1a, F5a, ...). Inputs name gen-7 content-table rows (`family/row#field`) that exist now (asserted
by `tests/sim/test_constructs.py`).

## Constructs (new behaviour behind a switch)

| Construct (stub) | Field: gen 5 -> gen 7 | What 0.7.2 does | Inputs from the gen-7 tables | Differential and pass bar | Phase |
|---|---|---|---|---|---|
| Hit interrupt, stun, flinch (`hit_interrupt.py`) | `hit_interrupt`: False -> True | 31 attack files parent `DamageEntityParent_Hit_Interrupt` (Weapons: Daggers 9, Mace 7, Sword 6, Battleaxe 5; Abilities: Fireball 1, GroundSlam 2, WindStrike 1). On hit, if the target carries `Hit_Interrupt_Eligible` and none of Flinch/Enraged/Recovery_Pending: cancel its attack, stun, flinch, record the reaction | `weapons/interaction:*#hit_interrupt`; `weapons/interaction:DamageEntityParent_Hit_Interrupt_On_Hit#document`; `weapons/effect:Hit_Interrupt_*#duration,document` (Flinch 0.5 s at speed x0, Stunned 0.7 s, Enraged 8 s, Recovery_Pending 2 s, Recent_Reaction_A/B 4 s, Eligible 1 s); eligibility: `npc_roles/role:*#component_references` reaching `Component_Instruction_Hit_Interrupt` along `reference_chain` | live hits on an eligible role (Skeleton_Fighter, Goblin_Scrapper): attack-cancel tick, flinch window and displacement, stun, enrage after the third flinch; match native within tolerance (F5a). Note F5a: the current opponents (Trork_Brawler, Trork_Unarmed, Kweebec_Razorleaf) are not eligible | P3 |
| Mana pool (`mana.py`) | `mana_pool`: False -> True | Mana 100/100 at spawn, +1 per 0.3 s while alive (0.5.10: 0/0, +1 per 0.2 s gated on 6 s without damage and not charging) | `stats/stat:Mana#initial_value,max,regenerating` | refill rate +1 per 0.3 s within 5 % (F7a) | P3 |
| Rune casting (`mana.py`) | `rune_casting`: False -> True | Rune items carry an `Ability` block (slot, cost in mana, cooldown, keywords, `Cast` root); casts launch ability projectiles (`LaunchAbilityProjectile`) | `weapons/item:Rune_*#ability` (e.g. Rune_Fireball cost 30, cooldown 2); `weapons/root:Root_Ability_*`; `weapons/interaction:Ability_*_Cast#document` (fireball cast RunTime 0.87); `weapons/projectile_config:Projectile_Config_Ability_*`; modifier runes need `Server/Abilities` keyword semantics (not extracted) | per rune: cost, cooldown, cast time, flight and effect match native (F7b) | P3 |
| Hookshot (`hookshot.py`) | `hookshot`: False -> True | Tool_Hookshot fires, attaches and pulls; Rune_ChainHook casts a chain hook | `weapons/item:Tool_Hookshot` and its graph; `weapons/projectile_config:Projectile_Config_Hookshot`, `Projectile_Config_ChainHook`; `weapons/interaction:Ability_ChainHook_Cast`, `_Hit` | fire, attach, pull distance and timing (catalogue: chain hook to 1.5) | P3, on demand |
| Dodge action head (`dodge.py`) | `dodge_action_head`: (left, right, forward, back) -> (left, right) | Nothing new: the seven dodge assets are identical in 0.5.10 and 0.7.2 and the engine already executes left/right only (`DODGE_AUTHORED_ACTION_MASK`); only the policy head shrinks | `weapons/interaction:Dodge#document` (only Left/Right branch to a chain), `Dodge_Left`/`Dodge_Right` (cost 2, force 13 Set, regen delay Set -0.7), `weapons/effect:Dodge_Invulnerability#duration` 0.25 | existing dodge boundary tests on 0.7.2 (F9 MATCH) | P3 |
| Line of sight (`line_of_sight.py`) | `line_of_sight`: non_transparent_blocks -> detail_hitbox | A non-transparent block occludes only where the ray hits one of its detail hitboxes (0.7.2 `PositionCache`); full cubes agree (F2a) | `blocks/block:*#opacity,hitbox_type`; `blocks/hitbox:*#boxes`; cell rotation from the Region capture | occlusion fixture with slabs, stairs, fences: every ray agrees (F2b); then the catch rule and the `hidden` opening | P2 |
| Slope and wall limits (`slope_limits.py`) | `slope_limits`: False -> False (decided 2026-10-18) | `MaxSlopeAngleDegrees` 30, `MaxWallAngleDegrees` 15 (new keys) with no 0.7.2 reader (jar scan: `MovementConfig` and `MovementSettings` only) | `movement/config:Default#fields.MaxSlopeAngleDegrees,fields.MaxWallAngleDegrees` (rendered as data) | none while no reader exists; a slope fixture (<= 5e-4 m per step, F1d) only if a reader is found | P2 |
| Walk descent (`walk_descent.py`) | `walk_descent`: free_fall -> minimum_descent_rate | 0.7.2 walk controller forces a minimum descent rate when descending (code) | none (engine code) | stairs/ledge/slope fixture extending `adk/tools/native_motion_differential.py`, <= 5e-4 m per step (F1e) | P2 |
| Climbing (`climbing.py`) | `climbing`: False -> True | Climb on `IsClimbable` blocks (present in 0.5.10 too; the gen-5 engine never models it) | `blocks/block:*#climbable` (57 resolved types; 44 files author the key); `movement/config:Default#fields.ClimbSpeed,ClimbSpeedLateral,ClimbUpSprintSpeed,ClimbDownSprintSpeed` | ladder fixture: up, down, lateral speeds (F10a) | after P2 |
| Gliding (`gliding.py`) | `gliding`: False -> True | Glider item (0.7.2 `Glider_Scrap`) | `weapons/item:Glider_Scrap`; `stats/stat:GlidingActive`; `stats/stat:Stamina` Gliding drain rule | glide speed, fall rate, stamina (catalogue 10, <= 2/s, 1/s) (F10d) | on demand |
| Riding (`riding.py`) | `riding`: False -> True | Mounts (`MovementConfig/Mount.json` new in 0.7.2) | `movement/config:Mount#fields`; mount roles in `npc_roles` | rider speed, sprint, jump (catalogue 8, 13.2, 14) (F10e) | on demand |
| NPC set-velocity (`npc_velocity.py`) | `npc_velocity_model`: zero_then_add -> role_set_velocity | `Role.processSetVelocityInstruction` applies one set velocity (~23x for set-type NPC knockback); never reaches the bridge-driven agent | none (engine code) | NPC-on-NPC knockback displacement | P3 |
| Dagger swing damage (`dagger_damage.py`) | `dagger_swing_damage`: shared_rows -> gen7_table | Swing_Left and Swing_Right of every dagger material rise (Iron 4 -> 6, Bronze 3 -> 5, Mithril 8 -> 12, Thorium 5 -> 8, ...); the table's other direct values and the angled values are unchanged; Void_Crystal has no simulator row | `weapons/item:Weapon_Daggers_*#damage` (`Swing_Left_Damage`, `Swing_Right_Damage`, base_first) | the table check per variant and the seeded bronze-dagger contact, native 5.0 (F5c) | P3 (phase 4) |
| Guard cancel window (`guard_window.py`) | `guard_cancel_window`: guard_root -> attack_root | The guard roots of Sword, Mace, Battleaxe and Daggers lose `Interrupting: [Primary]`; each of their Primary attack roots is a `Simple` interaction with `InterruptedBy: [Secondary]` and a wind-up `RunTime` (sword swings 0.111 s, sword down 0.361 s, mace 0.536 s, battleaxe 0.348 / 0.4 s, daggers 0.066 s), so Guard cancels an attack only while its root runs | `weapons/interaction:<attack>` Type, RunTime, Rules | `tests/fidelity/combat/defense/test_native_guard_attack_cancel.py` (live): onset 4 cancels, onset 8 does not, on native and on the engine; per-attack windows measured 2026-10-18 (sword left 2-5, down 2-12, mace 2-18, battleaxe down-left 2-12) | 5d (2026-10-18) |
| Native binding types (`native_bindings.py`) | `native_binding_types`: profile_sets -> item_interactions | 0.7.2 binds the crossbows' reload root `Root_Common_StatAmmoReload_Entry` under `Interactions.Ability4` (0.5.7: `Ability3`); the bridge requests a slot by its binding's type name, and the native evidence advertises the slot only while the bridge reports the same name. The binding takes the item map's type where the map binds the root under one other type: of every profile's bindings only the two crossbow reloads change; the engine keeps the reload's ordinal | `weapons/item:<item>#interactions` | the crossbow reload and crossbow projectile tests on the live bridge (jar 32EF5EDD): the reload slot is advertised and the Ability4 reload runs; `tests/sim/test_native_bindings.py` (only the two reload bindings change) | 6c (2026-10-18) |
| Bench crafting (`bench_crafting.py`) | `bench_crafting`: fieldcraft_context -> recipe_bench | Not a 0.7.2 change: the engine commits every Region craft under the Fieldcraft context, so a recipe offered because an ordinary bench is in reach is refused at the commit (2 of 197 offered 0.7.2 recipes commit; gen 5 keeps this defect by ruling), and more than 16 legal recipes fail the actor row closed while the native bench window lists them all. Gen 7 commits under a bench the recipe names and keeps the first 16 legal recipes in table order | the packed recipe table's bench requirements (native catalog recording) | `tests/sim/test_bench_crafting.py`; a native bench-craft differential (open) | 11, stage 0 (2026-10-18) |
| Crafting materials (`crafting_materials.py`) | `crafting_materials`: concrete_only -> resource_types | A 0.7.2 recipe input can name a resource type ("any 4 Wood_Trunk"), and the server excludes the recipe's own outputs from it (`applyOutputExclusions`); the Region action admitted only concrete inputs and its inventory published no resource types. Gen 7 admits a resource input that excludes exactly its recipe's outputs, publishes each slot's resource types from the archive's item `ResourceTypes`, and never lets such an input consume the recipe's output | the archive's `Server/Item/Items/**` `ResourceTypes` (read at compile time) | `tests/sim/test_crafting_materials.py`; the 0.7.2 catalog count (263 offered, 263 commit) | 11, stage 1 (2026-10-18) |
| NPC walk jump velocity (`npc_jump.py`) | `npc_walk_jump_velocity`: player_force -> native_npc | HEAD gives the NPC learner (`npc_walk`) the player's `PLAYER_JUMP_FORCE` 11.8 (apex 2.18 m); the 0.7.2 NPC walk controller's first jump step is 5.099 m/s = sqrt(2 x 10.0 x 1.3), the NPC motion parameters being identical to 0.5.10 (TRAINING item 6); measured 5.09901951359 on the bridge; since the engine batch of 2026-10-18 (late) gen 5 computes the same formula for `npc_walk`, so the construct changes no value on that HEAD | `rulesets/hytale_0_5_7` `agent.jump_velocity_gravity_floor` 10.0 and `agent.jump_height_parameter` 1.3 (player model unchanged) | native first-step vertical speed and `native_motion_differential --jump-steps 1`, <= 5e-4 m per step (F1c) | P2 (phase 3b) |
| NPC jump arc (`npc_jump_arc.py`) | `npc_jump_arc`: constant_gravity -> physics_math | The 0.7.2 NPC walk controller carries a jump as a fixed applied velocity plus an external speed that `computeNewFallSpeed` updates after each move: `e += -32 (1 - |e / T|^3) dt`, so the vertical speed on airborne step k is `v_(k-1) - 32 (1 - |(v_(k-1) - v_jump) / T|^3) dt_(k-1)` (the previous tick's duration and a cubic drag), `T = PhysicsMath.getTerminalVelocity(mass, 0.001225, hitbox width x depth, drag)` = 140.488 m/s for a model without `PhysicsValues` (the `ModelAsset` default mass 68, drag 0.5) and the 0.6 x 0.6 learner hitbox; the engine's legacy branch integrates `v_(k-1) - g dt_k` | the agent bounds (`rulesets/hytale_0_5_7` agent bounding box); mass and drag are `ModelAsset` code defaults (no asset authors them for the learner model) | `native_motion_differential --jump-steps 1`, <= 5e-4 m and 1e-3 m/s per step; the formula reproduces four live jump-in-place captures to 1e-13 (double precision); `tests/sim/test_npc_jump_arc.py` (F1c). With it, the overlay patches `npc_rejump_import` and `npc_rejump_after_landing` lift the engine's landing-recovery jump deferral for `npc_walk` (natively a held jump fires on the tick after landing) | 1c (2026-10-18) |
| Block health reset (`block_health.py`) | `block_health_reset_on_write`: False -> True | health entry dropped once health leaves (0, 1) and on any block write | none (engine rule) | mutable-block native compare re-baselined on 0.7.2 | P5 |

`doors` and `swimming` keep their gen-5 models in gen 7 (`world_v0_5`, `fluidfx_v0_5`) until a 0.7.2
differential differs; the doors source audit crashes on 0.7.2 on removed classes (F10b, F12) and the
swimming compare asserts 0.5.7 (F10c), so both need P5 re-baselining before they can measure.

## Content changes that need no construct (the overlay's redirects and patches)

These are new numbers flowing through behaviour the engine already has. They become redirects (a
symbol reads its table field) or, where the gen-5 engine encodes the number structurally, a
fidelity patch in the gen-7 overlay. Every value below is in the gen-7 tables and asserted by
`tests/sim/test_a2_diff.py`.

| Change | Table field | Engine target today |
|---|---|---|
| Sword knockback: Swing_Left 6.0 at (0,1,-1) -> 4.3 at (0,0.2,-1); Swing_Right 7 -> 5.0; Swing_Down 12 at (0,1,-2) -> 10.8 at (0,0.2,-2) | `weapons/interaction:Weapon_Sword_Primary_Swing_*_Damage#knockback` | `profiles/hytale_0_5_7.py` force tuples (redirect `sword_knockback`) |
| Sword hold gates 0.2 / 0.65 -> 0.174 / 0.565 (not in TRAINING-0.7.2.md; a lead for F5f, the agent's own hits landing 1-2 ticks later) | `weapons/interaction:Weapon_Sword_Primary#charge_thresholds`, `Weapon_Sword_Primary_Thrust#charge_thresholds` | `_NESTED_OUTER_GATE_SECONDS`, `_SWORD_THRUST_HOLD_SECONDS` (redirect `melee_hold_constants`) |
| Dagger swings +50 % (Iron 4 -> 6, Mithril 8 -> 12) | `weapons/item:Weapon_Daggers_*#damage` | `_iron_daggers` defaults, `SCALAR_VARIANTS` |
| Ranged: Shortbow Iron 6/9/12/16 -> 1/2/4/7, launch 15/30/50/70 -> 12/22/35/50 (full draw 85/25 unchanged); crossbow Iron combo 27 -> 13, Ancient Steel 42 -> 14 / Big Arrow 123 -> 82 / standard 16 -> 10.5 | `weapons/item:*#damage`; `projectiles/config:Projectile_Config_Arrow_Shortbow_Strength_*#launch_force` | `shortbow_tiers`, `iron_template_damage`, `SCALAR_VARIANTS` redirects |
| Durability: 57 of 143 compiled rows change maximum, 82 change loss (Sword Iron 120 -> 250, 0.21 -> 0.13) | `durability/item:*` | `hytale_0_5_7_item_programs_v1.json#weapon_items` (redirect `compiled_weapon_durability`) |
| Guard roots no longer interrupt Primary (`Rules.Interrupting: [Primary]` removed from the Sword, Battleaxe, Daggers and Mace guard roots) | `weapons/root:Root_Weapon_*_Secondary_Guard#document.Rules` | `_guard(interrupting_type_mask=1 << PRIMARY)` default (patch: read the root's Rules) |
| Crafting 1,947 -> 2,047 recipes, 22 bench ids, 19 without a bench | `crafting/recipe:*` | `jax/crafting/contract.py` `RECIPE_TABLE_CAPACITY` (redirect `crafting_capacities`) |
| Spider web speed 0.65 -> 0.45 (jump 0.5 -> 0.45, drag 0.45 -> 0.82) and a new NPC web-collision slow (x0.2, 1 s) | `blocks/block:Deco_SpiderWeb#movement_settings,collision_interactions` | Region recapture on 0.7.2 (the engine reads captured cells) |
| Mana 0 -> 100 | `stats/stat:Mana` | `mechanics/factory.py` defaults (redirect `mechanics_factory_resources`) |

## Engine fixes that are not version constructs

- F1c NPC jump: native gravity integrates with the previous tick's duration, the engine with the
  current one (inferred from receipts); it failed on 0.5.9 too. A fix belongs in the engine for every
  generation, gated by the gen-5 identity check, not behind a gen-7 switch.
- F5f the agent's own hits land 1-2 ticks later on 0.7.2: cause not isolated; the sword gate change
  above is the first lead to test.

## Status after phase 3 (2026-10-18)

- **Line of sight (hook 2b): implemented.** `mechanics/line_of_sight.py` tests each opaque cell's detail
  boxes (`GeometryState.collision_boxes`) and keeps the gen-5 cell rule for opaque cells that carry no
  boxes. The full-cube rays agree with native on the live bridge under both rules
  (`FIDELITY-0.7.2.md` section (g), row 2a). The partial-box differential has no runnable fixture, so
  row 2b is unmeasured.
- **Slope and wall limits (`slope_limits`): decided 2026-10-18, switch False (phase 3b).** The 0.7.2 server jar reads neither
  `MaxSlopeAngleDegrees` nor `MaxWallAngleDegrees` (scan of 38,436 class files; the names appear only in
  `MovementConfig` and the client struct `MovementSettings`). The engine's contact normals are
  axis-aligned, so no angle threshold can change a contact. Decision (owner ruling carried into phase 3b,
  recorded 2026-10-18): option (a), the switch set False, so the gen-7 server matches the gen-5 server on
  this field. Not taken: a documented no-op switch, and a player-body measurement (no player-client oracle
  on the bridge).
- **Walk descent (`minimum_descent_rate`): not hooked, premise unverified.** TRAINING-0.7.2.md item 8 states
  it from code reading; a static check of the walk controller's fall-speed method found the 0.5 min/max
  form only. The whole walk update was not checked.
- **Step-up and climbing: no engine branch.** `jax/combat/runtime/agent.py` has no climb move. The 0.7.2
  NPC walk controller has `computeClimbMove` and `computeClimbSpeed` (javap). A 0.5 m slab is not climbed
  on either side (`half_block` probe: y 65.0 and z -0.5 after 60 steps). Stairs are unmeasured.
- **Jump impulse (row 1c): decided in phase 3b; gen 7 uses the native NPC value** behind `npc_walk_jump_velocity`
  (`npc_jump.py`; gen 5 keeps `player_force`). The differential result is in FIDELITY-0.7.2.md (g). Earlier text:
  The committed NPC learner jumps with
  `PLAYER_JUMP_FORCE` 11.8, where native NPCs jump with 5.099. The uncommitted working tree already uses 5.099.
  Measured on HEAD: 2.38 m error (gen 5), 2.42 m (gen 7). Not changed here.

