# tasks/ — the composition task and its arenas

The one task every middle-layer instance is measured on: pursue, evade when damaged, resume pursuit
(ADR-001 P2). It exists to measure composition, not skill quality: the skills are the stand-ins of
`../skill_library/`.

| File | Lines | What it is |
|---|---:|---|
| `composition.py` | 421 | `CompositionTask`: the executive state it runs under (pursuit always, evader with the `on_damage` constraint label, `EVADER_PRIORITY` above `PURSUIT_PRIORITY`), the per-episode verdict `classify` (composed, composed_loose, pursue_only, failed), the tick budget. |
| `arenas.py` | 510 | Arena selection for the runs: one builder, four arenas (`combat_open_flat` is the standard; `combat_cover` is the pillar arena below). |
| `flat_arena.py` | 688 | The flat two-actor pursuit arena, one decision at a time. |

Gates: `test_na_tasks.py`, `test_na_arenas.py`.

Removed 2026-10-18 (owner-approved cleanup): `events.py`, the damage-event read helper and ablation injector nothing on the current path used.

Known measurement facts: `j0` on seed 0 at 512 ticks, batch 8, 8 episodes is 43/81 = 0.531 (Wilson lower
0.423); catches depend on the horizon (1 tick = 0.045 s in the gen-5 simulator; the gen-7 engine steps at
1/30 s, so horizons will be restated in ticks when the port lands).

## Arena `combat_cover` (task #98, 2026-10-20)

`build_arena("combat_cover", batch, ticks, seed, loadout=None, arrows=None)` in `arenas.py`; `--arena combat_cover` in `eval/run_composition.py`.
It is the `combat_open_flat` legal scene (default reset, default melee opponent, 120-degree sector, sensor range 16) with local
geometry added through `geometry_arsenal_world_capabilities` and `geometry_provider`:

- Floor: solid and opaque over the whole 9 by 9 window about the spawn block (origin `(0, 65, 0)`).
- Pillars: six 1 by 1 columns, 3 high, at block offsets `(3,0) (1,3) (-1,3) (-3,0) (-2,-3) (2,-3)`, radii 3.0 and sqrt(13) from the spawn.
  Task #116 (2026-10-20) moved the two front columns one block outward, from `(-1,-3)` and `(1,-3)`. They boxed in the NPC's
  spawn: its box is 0.95 wide at centre `(0.5, -2.0)`, so its diagonal chase step collided, its navigation was rejected, and the
  sticky `target_navigation_unsupported` froze it there. Measured with `resolve_aabb_motion` on the cover geometry: 42 of 110
  straight steps (0.02 and 0.1 block) from the spawn toward the window's agent cells collided before the move and 0 after.
  The record (`describe()["pillars"]`) gives the offsets, radii and angles. The contract digest covers the layout, so the move
  changed it: the pinned digests in `test_na_arenas.py` were re-captured, and the pre-move layout reproduces the old values.
- Radius, measured on 2026-10-20 with `geometry_perception_line_of_sight_result` (`hytalegym/jax/world/perception/los.py`):
  the requested 5 to 15 blocks is beyond the local window (`hytalegym/geometry/contract.py`, `CELL_RADIUS = 4`). A clear
  line is visible at block offsets 1 to 4 and not at 5, where `geometry_exhausted` is set. The usable radius is 3 blocks
  on an axis (a pillar at 3 blocks hides a target at 4) and 4.24 (sqrt 18) on the diagonal (pillar (3,3) hides (4,4)).
- Window limit (measured 2026-10-20 with `resolve_aabb_motion` and `environment_damage_contact`): `geometry_exhausted` is sticky
  and ends the episode (`arsenal/runtime.py` 2240 and 2364-2368; `environment.py` publishes `done = terminated | truncated`). So an
  agent centre beyond about -3.7 or +4.7 blocks on x or z from the origin is a world error: `terminated` is set with both bodies
  at full health (the composition's `terminated_alive` column). On x the switch lies between -3.695 and -3.700 and between 4.695
  and 4.700; on z between -3.690 and -3.705 and between 4.695 and 4.705. So combat_cover is not an arena for kiting at range.
  Measured by the diagnosis lane, not re-run here: on combat_open_flat a same-speed chaser follows a retreating agent at 3.4 to
  3.9 blocks.
- The NPC moves only when it sees the agent and its straight step is unobstructed (`hytalegym/jax/combat/opponents/legacy/motion.py`,
  `_target_navigation_rejected`). `combat_cover` passes no `target_navigation_provider`, so any rejected step sets the sticky
  `target_navigation_unsupported`. Probe (gen 5, the diagnosis lane's `scripted_avoid.py`, 200 ticks, the agent backing away):
  before the move, 4 of 4 lanes had the flag set before the window exit, the NPC frozen within 0.05 block of its spawn. After the
  move, 6 of 16 lanes still set the flag (ticks 109 to 153) while the agent was inside the window: the NPC was about (-0.49, -2.66),
  its box a few hundredths of a block from the east face of the `(-2,-3)` column, with line of sight clear. Of the 16 lanes, 8
  left the window at tick 43, and 2 stayed inside without the flag (the NPC moved up to 3.5 blocks). Gen 7 (the `gen7.py`
  launcher, 16 lanes, the same driver): no flag, but every lane left the window by tick 43, so that run does not reach the chase.
  The pillar move does not make combat_cover a chase arena; a navigation provider for it is the open fix, not made here.
- Loadout: `iron_sword` (default) or `iron_shortbow`, through `build_arena(..., loadout=...)`, and `--loadout` on the CLI.
  The opponent's loadout is unchanged.
- Arrows (task #111, 2026-10-20): `build_arena(..., arrows=N)` seeds the agent's `Weapon_Arrow_Crude` stack with N at every reset,
  through `inventory_reset_provider` in the scene's environment kwargs (`_arrow_inventory` of `duel_arena.py`). The collectors'
  autoreset resets through the same call. `None` (default) keeps the bare inventory, which holds no arrows, so a shortbow tap
  spawns nothing. A positive N on `iron_sword` raises. An arrows build is named `neural_agent/cover_<loadout>_arrows<N>` (scene
  `..._legal_...`), because agent names are process-wide. CLI: `--arrows N`. Measured on gen 5 (2026-10-20): reset holds N,
  one tier-4 tap spawns at tick 42 and takes one arrow.
- Gate: `tests/experimental/neural_agent/test_na_arenas.py` (cover layout and digest, loadout, the behavioural case, and the NPC's
  free spawn step, task #116).

## duel_arena.py (2026-10-20, task #101, the skill layer's league ask L1)

`make_duel_arena(batch, (loadout0, loadout1), *, view_sector_full_angle_radians, legal_observation_sensor_range)` returns an arena
whose `reset(keys)` gives `(state, obs0, obs1, mask0, mask1)` and whose `step(state, factors0, factors1, keys)` gives
`(state, obs0, obs1, mask0, mask1, info, done)`: both entities take external int32[B, 12] factors each tick, each gets its
own learner v3 observation and bool[B, 124] mask from its own perspective, and the opponent's built-in AI is off
(`arsenal_runtime_config(..., opponent_controller_mask=zeros)`). Built from the multi-actor package's pieces
(`policy_actor_assignment`, `actor_first_arsenal_view`, `observe_policy_actors_flat`, `decode_policy_actor_factors_flat`,
`step_multi_actor_arena`) on the open-flat capabilities; no engine edit. The adapter computes entity 1's perception by
applying the fixture on entity 1's view (the fixture gates the agent row only), copies the flat floor's actor-world rows to
entity 1 (the engine zeroes them for non-agent rows), and seeds both heads from `entity_yaw`. No episode horizon: the caller
stops the loop. Measured (tests/experimental/neural_agent/test_na_duel_arena.py, on disk): each entity moves under its own
factors, neutral factors keep both still for 100 ticks with the AI off, a sword tap damages entity 1 and both views track it,
the sector hides a target behind either entity, the sensor range hides both, masks are per entity. The stock open-flat scene has
no geometry window (geometry_exhausted stays False at 8.5 blocks): the 4-block window of the bow recipe comes from the tests'
exact flat geometry. Entity 1's health fractions and its combat-block forward and right velocity are rescaled to its own maxima
(`_entity_one_on_its_own_maxima`: self and combat agent health by target_max_health, the target row and the combat visible
target health by agent_max_health with the encoder's clip and wire rounding, the velocities by target_max_speed; six columns,
checked against the engine's encoder with the same-shape agent and target parameters swapped). The remaining role-scaled
normalisers (gait speed, fall speed, chase speed, the attack table, drop height, the status rows' max health) have no
counterpart in CombatParams and stay on the agent's scale; they are listed in the lane's DUEL_RECIPE.txt under G1 RESIDUAL, and
a target-role gait maximum in the engine's actor-first view is the fix.

### Roles and measured gaps (task #103b, 2026-10-20)

`make_duel_arena(..., roles=(entity 0 role, entity 1 role))` sets each slot's statuses (`entity_role_ids`) and rebuilds
the fields in `ROLE_PAIRED_FIELDS` so that a slot playing the target role takes the `target_X` value. The residual and
the physics caveat are in the function docstring. `roles=None` is bit-identical to the matchup (reset and 64 steps,
checked against the pre-edit code). Measured on gen 5 (not in the tests):

- A role pair does not give a slot its own movement. The motion kernel runs every entity row through the agent slot
  (`multi_actor/locomotion.py`, `_materialize_entity_zero`), so a walking body follows `agent_max_speed`. Entity 1 walks
  0.331 blocks in 8 ticks under the default and under (Kweebec, Kweebec), and 0.391 under (Trork, Trork) and under
  (Trork, Kweebec): slot 1 playing the Kweebec moves at the Trork's speed.
- The built-in target still moves entity 1, so entity 1 is not inert under neutral factors. Its motion
  (`opponents/legacy/motion.py`, `_tick_target_motion`, called from `runtime/step.py` and applied where
  `params.target_active`, default True) starts at the reset-drawn `target_ai_activation_tick` (36 to 44 in the ruleset).
  With neutral factors and entities 6 blocks apart, entity 1 first moves at tick 41 (key 0) and drifts 0.652 blocks in
  100 ticks; with `target_active=False` the drift is 0.000 and entity 0 does not move at any separation. The
  opponent-controller mask does not gate this motion. Candidate one-line fix, not applied here: `target_active=False`
  in the duel's default params. It changes the follow-up hits: with it, the sword-1 third hit lands in 4 of 16 seeds
  and the mace-1 second hit never lands.
- Knockback responds by role in the engine, not by parameter. The agent slot takes the force-pushed branch
  (`runtime/agent.py`, `force_pushed`, set only for damage to the agent in `arsenal/runtime.py` near 2563), the target
  slot takes the applied-velocity path (`mechanics/runtime/kernel.py`, `apply_damage_forces`). `target_knockback_*` is
  read only by the legacy opponent attack and does not change any hit tick (16 seeds, sword and mace).
- The one-tick first-hit lag of sword 1 is joint selector geometry. The clock, the acceptance tick and the head yaw are
  identical in both lanes. The progressive selector takes its origin from the source's eye offset (`events.py`,
  `_progressive_selector_result`: `agent_eye_offset` for slot 0, `target_eye_offset` for slot 1) and tests the target's
  box (`agent_bounds` or `target_bounds`) in `horizontal_selector_intersects_aabb` (`programs/selectors.py`). Tick 7
  needs origin 1.0 with the Trork box; every other combination tested gives tick 8.
- The follow-up hits depend on reset draws, not per-tick draws or the damage jitter: the activation tick above and the
  motion-timing profile (`runtime/reset.py`, the timing draw next to the activation draw). Without knockback there is no
  seat difference and no key dependence.

### Arrows for the shortbow (task #109, 2026-10-20)

`make_duel_arena(..., arrows=(entity 0 arrows, entity 1 arrows))` seeds each entity's arrow stack at reset. The reset's inventory is
`training_inventory_from_config(config)` with the quantity of every slot holding `ARROW_ITEM` (`Weapon_Arrow_Crude`, semantic ID
1135244191, the shortbow's per-shot removal of one arrow) set to that entity's count, and it is passed as the reset's
`initial_inventory`. The inventory is `[batch, entity, container slot]`; a shortbow loadout holds its arrow stack in one slot per
entity (slot 41 of 76 in the probe). The count is finite: 1 fires once and a second tap spawns nothing. The seam is
`initial_inventory` of `reset_arsenal_batch`, which `make_arsenal_environment` fills from its `inventory_reset_provider`; the duel's
reset takes no provider. `None` (the default) keeps the bare factory inventory, which holds no arrows, so a shortbow tap is admitted
and spawns nothing. An entity with a positive count whose loadout holds no arrow stack raises `ValueError`.

The tests (tests/experimental/neural_agent/test_na_duel_arena.py, on disk) check tier 0 at 2.5 blocks: entity 0's tap
spawns on tick 9 and hits entity 1 on tick 13 for 6, the bow recipe's ticks (0-based, tap at tick 0); the mirror spawns
on 9 and its hit is asserted within one tick of 13 (it lands on 14). A probe run after the edit, not part of the tests,
gave the mirror hit on 14 and, with 64 arrows, two taps spawning on 9 and 49 and hitting on 13 and 53.

Caveat: the shortbow's range per tier (the spawn and hit ticks of tiers 0 to 4, and how far a shot reaches) is the skill layer's
measurement, not this arena's. The tests here check tier 0 at 2.5 blocks only.

### Entity 1's attack facing (task #112, 2026-10-20)

The engine's progressive attack selector aims slot 0 with `combat.yaw` (the body) and slot 1 with `combat.target_head_yaw`
(`arsenal/effects/events.py:809-821`). Only the built-in target AI writes that leaf, and it is off in the duel, so the leaf
kept its reset seed while entity 1's body turned, up to 58.7 degrees off. `step` now sets it to entity 1's body yaw
(`_target_head_on_body`) on the state it receives and on the state it returns. No lag at the selector: the engine
resolves attacks before either body turns that tick (`arsenal/runtime.py:1302` before `:1519`, entity 1's turn after
the engine step), so both slots aim with their start-of-tick body yaw. Pitch is not synced; both seats' selector pitch
is level for the whole duel.

Measured (gen 5, 64 lanes, seed 0, 600 ticks, roles (Trork_Brawler, Trork_Brawler), knockback 0). Sword PRESSURE mirror:
entity 0's dealt difference +21.2 ± 3.2 before, +11.0 ± 2.5 after. Mace HIT_AND_RUN with the seats exchanged: pairs won
by the same seat both ways 23 before, 1 after. Both match an in-process engine patch giving slot 1 its body yaw, lane by
lane. Fixed-placement held probes (one seat presses, 200 ticks) are unchanged in both seats (sword hits on 7, 20, 33, ...;
mace 21, 57, 93). The rest of the seat gap is engine-only: an agent-only control lock on any damage to entity 0 and
agent-only regeneration (task #108).

Caveat (read from the code; arrows not measured): the same leaf sets entity 1's projectile muzzle (`arsenal/environment.py`,
`muzzle_yaw`), and its pitch reads the level `target_head_pitch`. Entity 1's arrows therefore leave along its body yaw,
level, where entity 0's leave along its look (before this change they left along its spawn heading). Shortbow numbers
from entity 1 are not valid; shoot from entity 0. Entity 0's view of entity 1's yaw (the legacy path in `multi_actor/views.py`) now shows the body yaw too.

### Two engine opt-ins: the control lock and entity 1's regeneration (task #108, 2026-10-20)

`make_duel_arena(..., knockback_gated_control_lock=True, target_regeneration=True)` turns both on, as the duel's default. With both `False` the duel is the stock engine bit for bit: `test_opt_ins_off_leave_the_stock_duel_bit_for_bit` compares every output leaf of three probes (knockback 1 tap, knockback 0 freeze, a 480-tick regeneration probe) with digests recorded on the stock engine before the change. Single-agent arenas do not pass the keywords and keep the stock path.

- **F2, the lock follows a push.** The engine locks the agent's control on any damage: `target_damage_applied_this_tick` is set by `_resolve_damage` (`arsenal/runtime.py`) and read by the agent locomotion as `force_pushed` (`runtime/agent.py`). With the opt-in, `ArsenalWorldCapabilities.knockback_gated_control_lock` (set per batch on the duel's commands) ANDs that flag with `agent_pushed`: the agent's applied knockback velocity read after the direct damage packet has applied its push (`arsenal/runtime.py`, `lock_input`, just before `_microtick`). A hit locks only where it pushes. At knockback 0 no hit pushes, so nothing locks: the stock freeze after the first hit is gone.
- **F3, entity 1 regenerates.** The engine heals the agent slot only (`runtime/step.py`, `_tick_regeneration`). `tick_target_regeneration` (`runtime/step.py`) applies the same rule to entity 1: `regen_fraction` of `target_max_health` every `regen_interval_ticks` after `regen_delay_ticks` without damage, while alive and below maximum. The duel calls it after each tick with the multi-actor locomotion's `ticks_since_damage[:, 1]`, which already counts that tick. The two bodies share a role, so the same rule is the consistent choice; the regeneration fields have no target-prefixed counterparts. Native NPC regeneration is not verified.
- **Why a world-capability field and not a config field.** A new `ArsenalRuntimeConfig` field would enter the checkpoint provenance digest (`training/multi_actor/checkpoint/provenance.py`, `arsenal_runtime_config_content_sha256` over every leaf, line 50), and a new `CombatParams` field would enter the Region reset filter digest (`training/region_resets.py`, `_combat_params_sha256`, recorded in `artifacts/region-combat-reset-pool-*`). Either would move those digests for the default config. Threading a flag through `step_multi_actor_arena` would need an edit to `training/multi_actor/runtime.py`, which this lane did not touch.
- **Not changed.** The target's own lock: its `force_pushed` comes from the multi-actor locomotion (`training/multi_actor/locomotion.py:537`). The pending-hit path (`_defend_pending_target_damage`) is not reached in the duel, because the legacy target attack is gated on `target_active`, which the duel sets False. The effects engine (`effects/runtime/engine.py`) is not on the duel path either.

Measured (gen 5, 64 lanes, seed 0, 600 ticks, roles (Trork_Brawler, Trork_Brawler), seats exchanged, scratch harness `lane_108/seatrun.py`). Dealt difference (entity 0 minus entity 1 in each run, ± SE). Knockback 1 is the spec; knockback 0 is the extra table, because F2 is inert when every hit pushes.

| mirror, knockback 1 | stock, run A / run B | both opt-ins, run A / run B | F2 only, per lane |
|---|---|---|---|
| sword PRESSURE | −0.42 ± 2.40 / +4.05 ± 2.18 | −4.33 ± 2.17 / −0.48 ± 2.03 | 0 of 64 lanes change |
| sword HIT_AND_RUN | −9.41 ± 2.36 / −5.92 ± 2.39 | unchanged | 0 of 64 |
| mace PRESSURE | −7.09 ± 2.25 / −8.45 ± 2.14 | unchanged | 0 of 64 |
| mace HIT_AND_RUN | −6.83 ± 2.20 / −8.14 ± 1.99 | unchanged | 0 of 64 |

Only F3 moves a knockback-1 bout: the 600-tick sword PRESSURE mirror (29 of 64 lanes change), where long undecided bouts reach the 450-tick delay.

| mirror, knockback 0 | stock, run A / run B | both opt-ins, run A / run B | F2 only, run A / run B |
|---|---|---|---|
| sword PRESSURE | +11.00 ± 2.50 / +12.98 ± 2.30 | −1.25 ± 2.19 / +1.25 ± 2.19 | +11.00 ± 2.50 / +13.34 ± 2.30 |
| sword HIT_AND_RUN | −2.23 ± 1.99 / −0.52 ± 2.06 | +0.03 ± 2.20 / −0.03 ± 2.20 | +0.03 ± 2.20 / −0.03 ± 2.20 |
| mace PRESSURE | +1.00 ± 1.00 / −1.19 ± 1.00 | +1.14 ± 1.01 / −1.14 ± 1.01 | +1.14 ± 1.01 / −1.14 ± 1.01 |
| mace HIT_AND_RUN | −0.41 ± 0.51 / −1.64 ± 0.86 | +1.14 ± 1.01 / −1.14 ± 1.01 | +1.09 ± 1.01 / −1.14 ± 1.01 |

At knockback 0 the stock duel gives the same sign in both seat orders on sword PRESSURE and on both HIT_AND_RUN mirrors; with both opt-ins each pair has opposite signs. F2 alone gives the both-on values to within 0.05 on sword HIT_AND_RUN and both mace mirrors. F3 (the gap between the F2-only and both-on columns) is what changes sword PRESSURE, where the 600-tick bouts reach the 450-tick delay.

On an engine built from HEAD (no `knockback_gated_control_lock` field and no `tick_target_regeneration`), both opt-ins are no-ops: `_opt_in_engine` in `duel_arena.py` resolves the pieces when a duel is built, so the module imports, and `test_duel_builds_and_steps_on_an_engine_without_the_108_pieces` pins that path.

Slot 1 knockback (task #108, 2026-10-21): `target_force_pushed_knockback` (`True` by default in the duel) lets a hit that damages and pushes entity 1 take its multi-actor locomotion row's force-pushed branch, as entity 0 takes it, and holds its body yaw under applied motion (`multi_actor/runtime.py` `_force_pushed_rows` and `_applied_motion_rows`, `multi_actor/locomotion.py` `tick_entity_policy_locomotion`); `False` is the stock path bit for bit.
