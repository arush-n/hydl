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

`build_arena("combat_cover", batch, ticks, seed, loadout=None)` in `arenas.py`; `--arena combat_cover` in `eval/run_composition.py`.
It is the `combat_open_flat` legal scene (default reset, default melee opponent, 120-degree sector, sensor range 16) with local
geometry added through `geometry_arsenal_world_capabilities` and `geometry_provider`:

- Floor: solid and opaque over the whole 9 by 9 window about the spawn block (origin `(0, 65, 0)`).
- Pillars: six 1 by 1 columns, 3 high, at block offsets `(3,0) (1,3) (-1,3) (-3,0) (-1,-3) (1,-3)`, radii 3.0 and sqrt(10) from the spawn.
  The record (`describe()["pillars"]`) gives the offsets, radii and angles; the contract digest covers the layout and the flags.
- Radius, measured on 2026-10-20 with `geometry_perception_line_of_sight_result` (`hytalegym/jax/world/perception/los.py`):
  the requested 5 to 15 blocks is beyond the local window (`hytalegym/geometry/contract.py`, `CELL_RADIUS = 4`). A clear
  line is visible at block offsets 1 to 4 and not at 5, where `geometry_exhausted` is set. The usable radius is 3 blocks
  on an axis (a pillar at 3 blocks hides a target at 4) and 4.24 (sqrt 18) on the diagonal (pillar (3,3) hides (4,4)).
- Consequence (read in `hytalegym/jax/combat/arsenal/runtime.py` 2240 and 2364-2368, not run): `geometry_exhausted` is sticky
  and truncates the episode, so an actor that leaves the window ends the episode. Measured only for a target placed at offset 5.
- Loadout: `iron_sword` (default) or `iron_shortbow`, through `build_arena(..., loadout=...)`. The CLI has no loadout flag, so
  composition runs use the default. The opponent's loadout is unchanged.
- Gate: `tests/experimental/neural_agent/test_na_arenas.py` (cover layout and digest, loadout, and the behavioural case).

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
