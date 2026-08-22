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
