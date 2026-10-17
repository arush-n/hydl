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
