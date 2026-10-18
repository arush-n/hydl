# HytaleGym

`hytalegym` is the JAX environment and native-session package used by Arena,
ADK, and the concrete agents. It models the published combat and locomotion
surface in pure array code, while keeping native bridge communication behind an
explicit runtime boundary.

The package depends on JAX, NumPy, Gymnasium, and external Hytale asset/server
inputs for native or asset-backed operations. Higher-level packages depend on
it; the gym does not import Arena or ADK.

## Entry points

- [`hytalegym/envs/`](hytalegym/envs) exposes the Gymnasium-compatible
  environment boundary.
- [`hytalegym/jax/`](hytalegym/jax) contains combat, world, training, and
  evaluation implementations.
- [`hytalegym/geometry/`](hytalegym/geometry) defines geometry frames and
  collision/provider contracts.
- [`hytalegym/worldgen/`](hytalegym/worldgen) contains native and captured-world
  seams.

The repository shim is [`__init__.py`](__init__.py); the installable source is
the nested [`hytalegym/`](hytalegym) package.

## Legal observation switch (2026-10-17)

The arsenal runtime has a `legal_observation` flag on `ArsenalRuntimeConfig`, default `False` (decided 2026-10-17 after an isolated run of the combat tree: the legal default broke four tests that encode the privileged semantics, so existing consumers keep the old behaviour and the neural_agent arenas opt in through the scene builder's `legal_observation` runtime option).

- `True` (opt-in; the neural_agent arenas set it): the learner sees only what its actor may perceive. A `REQUIRE_LINE_OF_SIGHT` ability is legal only when the perception ray reaches a living target (the environment fills `ArsenalWorldCapabilities.selector_line_of_sight_valid` with that predicate; the schema is unchanged so trained checkpoints keep their contract hash), and the per-entity defense, resource and status rows of entities the actor cannot perceive, or that are not living, are blank. A living enemy behind a wall is therefore indistinguishable from no enemy (three-world negative control in `tests/jax/combat/arsenal/tests/agents/ppo/test_leakage_three_world.py`).
- `False` (default): the privileged behaviour from before this date, for agents that are allowed to cheat (for example a Learning by Cheating teacher): legality follows `selector_line_of_sight_valid`, which carries target existence, and every entity row is published. Set it with `runtime_config._replace(legal_observation=False)`; the test `test_privileged_mode_restores_the_leak` shows the mode still leaks, so the switch is real.

`ArsenalResetBatch` also accepts the per-entity form `entity_position` / `entity_health` (zero health marks an absent slot), so a world without an enemy can be built through the reset seam.

### View sector on the two-entity path (2026-10-18, opt-in)

`geometry_arsenal_world_capabilities(..., view_sector_full_angle_radians=angle)` is the switch. The default `None` keeps the certified pair unchanged: `target_candidate_perceptible` is line of sight only, so an opponent directly behind the agent is perceptible. A finite angle below tau (for example `2 * math.pi / 3`, a 120-degree full sector) asks for the sector. The sector is centred on the agent's head yaw (`CombatState.agent_head_yaw`, since perception is head-frame; `native_view_sector` takes no pitch), with the native sector test and the validity rules of the multi-entity branch. An engagement target outside the agent's sector is no target for the agent: its id becomes -1 and its mask False, the state an absent opponent gives. Line of sight, the line-of-sight ability gate, the projectile predicates and the area centre therefore see no opponent. The agent's entry in `target_candidate_perceptible` is gated too. No predicate is added. A full circle (`math.tau`, concrete or traced) keeps the certified bits, because `jnp.where` selects them. The opponent's own entry is never gated. The option does not depend on `legal_observation`: a caller who passes a finite angle asks for the sector in either mode. Tests: `tests/jax/combat/arsenal/tests/environment/test_pair_view_sector.py`.

Measured for four loadouts (iron_sword, iron_shortbow, iron_crossbow, flame_staff) at one offset (opponent 2.5 blocks behind the agent, `legal_observation=True`). With the 120-degree sector, the policy flat with the opponent behind the agent equals the no-opponent flat (0 differing entries; `test_policy_flat_hides_an_opponent_behind_the_agent_under_the_sector`). With no sector, it differs (`test_policy_flat_shows_an_opponent_behind_the_agent_without_the_sector`). On iron_sword the learner's `defense_f32` and `resource_f32` rows for the opponent become zero under the sector, and the target block matches a world with no opponent (`test_learner_defense_and_resource_rows_follow_the_sector`).

Consequences, opt-in only. The masked engagement target feeds the learner and the capability-gated actions: the line-of-sight ability gate (`programs/ability.py`), the line-of-sight events (`effects/events.py`), the projectile predicates, and the area centre that the area effects read (`effects/areas.py`, `world.area_center`). So under a sector the agent cannot use those actions on an opponent outside it. The masking applies whatever `legal_observation` is. `training/arsenal_evaluation.py` (`_world_capabilities_ready`, around line 784) requires `selector_line_of_sight_valid` on every row, so rows with an opponent behind the agent drop out of that filter, as rows with a wall already do (the same expression). Not reached by this change: the step's own engagement id on the two-entity path (`arsenal/runtime.py`, `decision_targeting`, around line 922) is computed from the state with no candidate evidence, so it never sees the mask; the encoder's own engagement id (`observation/v3/encoding/encoder.py`, around lines 166 and 219) is unmasked as well, although the flat measurement above shows no learner-visible effect for the four loadouts.

Not gated, by decision: the opponent's own entry (`candidate_perceptible[opponent, agent]`), which the NPC's opponent memory reads (`opponents/runtime/kernel.py`, `step_opponent_memory`, lines 64 to 78). The NPC's own native view cone is a separate fidelity item. It is unmeasured and not modelled here.

Head frame: the multi-entity branch keeps `combat.yaw` for the agent (`arsenal/environment.py`, line 2508, where the target takes `target_head_yaw`). The two-entity sector uses `agent_head_yaw`. The two agree after reset and differ only when the head turns away from the body. Follow-up, not changed here: on a turned head the two paths centre the agent's cone differently. The multi-entity branch, the certified path, is left as it is.

### Sensor range: the option, and which number applies (2026-10-18)

`geometry_arsenal_world_capabilities(..., legal_observation_sensor_range=True)` (default `False`; two-entity path only, `ValueError` otherwise) blanks the agent's entry for an opponent beyond `params.sensor_range`. The distance is the planar (x, z) norm between the bodies, the target block's measure (`runtime/reset.py` `target_evidence`, lines 617 to 621; `runtime/math.py` `_target_distance`, lines 75 to 80). The encoder reads the agent's entry under `legal_observation=True` only, so the option has effect only there. The open-field control fixture `open_flat_arsenal_world_capabilities` takes the same keyword, because the local LOS window (`geometry/contract.py`, `CELL_RADIUS = 4`) cannot carry a sight line beyond about 4 blocks. Also masks the engagement target beyond `params.sensor_range`, as the sector does, so a line-of-sight ability stays illegal beyond the range. Tested inside the local window (a 2-block range against an opponent at 2.5 blocks: `test_sensor_range_blocks_a_line_of_sight_ability_beyond_it_inside_the_window`). The 16-block case with a clear sight line is inferred, not tested: the local window stops at about 4 blocks and no fixture carries a longer sight line.

Channel: a perception keyword, like the view sector. The flag is in none of `ArsenalRuntimeConfig`, `CombatParams` or `ArsenalWorldCapabilities`, and the encoder call is unchanged, so no digest moves. Neither option is part of the checkpoint provenance digest: `arsenal_runtime_config_content_sha256` (`training/multi_actor/checkpoint/provenance.py`, lines 50 to 84) covers `ArsenalRuntimeConfig` leaves only. A run that turns either option on must record it in its own manifest.

The multi-entity minimum is a trap. On the multi-entity shape, targeting uses `min(config.targeting_rules.sensor_range, params.sensor_range)` (`arsenal/runtime.py`, `combat_target_selection`, lines 2324 to 2332, guard `sensor_range.shape[1] != ENTITY_COUNT`). On the two-entity path that minimum is skipped: targeting uses `config.targeting_rules.sensor_range` alone (the builder default is `inf`, `runtime.py` line 261). The target block and this option use `params.sensor_range` alone. On the two-entity path, set `params.sensor_range`; changing only the targeting rules does not move the target row.

### Pinned digests (checked 2026-10-18)

Before and after these changes the following are unchanged: `active_combat_observation_contract_sha256` (`combat/contracts/policy.py`, lines 48 to 66 and 110 to 113), and `arsenal_runtime_config_content_sha256` for the iron_sword default with `legal_observation` False and True. The options add no field, so no schema or contract hash moves. Re-checked after the engagement-mask change (2026-10-18): the same three values, and the two fingerprints.
