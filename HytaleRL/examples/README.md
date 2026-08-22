# HytaleRL examples

The examples are small command-line entry points for inspecting the gym,
training a policy, evaluating a checkpoint, and exercising the native bridge.
They are demonstrations of the public Python interfaces, not a second training
framework.

They depend on the nested `hytalegym` package and, for native commands, an
external Hytale server plus bridge installation. JAX-only examples can run
without a live server when their selected scene does not require native data.

## Where to start

- `random_agent.py` shows the smallest policy/environment loop.
- `jax_ppo_train.py` and `jax_policy_evaluate.py` show the lower-level JAX
  training/evaluation seam.
- `jax_arsenal_train.py` and the region examples show combat and captured-world
  paths.
- `capture_native_*.py` shows how native evidence enters the world boundary.

The supported project-level training workflow still goes through the Console;
see the root [`README.md`](../../README.md).
