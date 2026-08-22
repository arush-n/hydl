# Training algorithms

This folder contains the learning/update kernels. They operate on an already-defined scene, policy contract, and rollout format; they do not decide what a task means.

| File | Purpose |
| --- | --- |
| [`ppo.py`](ppo.py) | Scene-derived recurrent PPO configuration, initialization, collection, update, parameter warm starts, and reward relabeling. |
| [`evolution.py`](evolution.py) | JIT-compatible elitist evolution over arbitrary floating-point policy PyTrees. |
| [`auto_improvement.py`](auto_improvement.py) | Content-addressed paired train/evaluate/accept-or-rollback rounds. |
| [`__init__.py`](__init__.py) | Public algorithm exports. |

## PPO

`ppo_config()` derives observation/action shapes from the live scene rather than duplicating ABI widths. The PPO helpers preserve recurrent carry, support warm starts, and can relabel rewards/GAE after collection. The environment’s potential-shaping discount must agree with the PPO discount when the potential-invariance law is intended to hold.

## Evolution

The evolution state is a floating-point PyTree population. Row zero is the exact center/initial candidate, selection is elitist, crossover is uniform, mutation is bounded, and non-finite fitness rows are skipped. This is useful for small policy or hyperparameter searches, not a replacement for recurrent PPO.

## Auto-improvement

`AutoImprovementProgram` fixes the seed schedule, evaluation contract, and retention law. `run_auto_improvement()` keeps the whole optimizer/training state on the device and selects or rolls back with a tree-wide mask. A candidate is admitted only when it is finite, eligible, and improves the declared objective without violating retention constraints.

