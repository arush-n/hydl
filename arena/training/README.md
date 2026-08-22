# Arena training

`arena.training` is a collection of reusable JAX training machinery over Arena scenes and HytaleGym policy contracts. It is not a registry of trained agents. A caller supplies policy parameters, data, optimizer choices, and experiment budgets; these modules provide the environment seams, reward terms, contracts, and bookkeeping needed to run them reproducibly.

## The training path

```text
ArenaScene / collector
        |
        +--> training/contracts: lesson objective, scope, evidence
        +--> training/rewards: optional reward streams
        +--> training/skills: reset, target, causal signal kernels
        |
        v
training/algorithms: PPO / evolution / auto-improvement
        |
        +--> training/runtime: contracts, replicas, audits, tuning, monitor
        +--> training/selfplay: live duels or frozen leagues
        +--> training/imitation: demonstrations and native replay
        |
        v
arena.evaluation: held-out replicated promotion gate
```

## Folder map

| Folder | Purpose |
| --- | --- |
| [`algorithms/`](algorithms/README.md) | Recurrent PPO, PyTree evolution, and rollback-safe auto-improvement. |
| [`configs/`](configs/README.md) | Versioned example training configuration. |
| [`contracts/`](contracts/README.md) | Complete lessons: reward law, action scope, evidence, and horizon. |
| [`diagnostics/`](diagnostics/README.md) | Measurement-only timing and semi-Markov decision tools. |
| [`imitation/`](imitation/README.md) | Behavior cloning, native replay, projection, caching, and transfer. |
| [`rewards/`](rewards/README.md) | Composable reward streams, not standalone lessons. |
| [`runs/`](runs/README.md) | Concrete long-running training entry points. |
| [`runtime/`](runtime/README.md) | Training-run contracts, feature fusion, async evaluation, audits, tuning, and monitoring. |
| [`selfplay/`](selfplay/README.md) | Two-live-policy duels and one-learner frozen-opponent leagues. |

## Two meanings of “contract”

`training/contracts` defines the behaviour being taught: for example, pursuit or attack timing. `training/runtime/contracts.py` defines the training experiment: metrics, parameter domains, stop losses, feature versions, and environment identities. A reward term in `training/rewards` is smaller still; it can be mixed into a lesson but does not own an action scope or reset.

## Current implementation boundary

Pursuit is the runnable collector and concrete runner. The contract catalog also contains drafted lessons that are not yet wired into a collector. Learned reward models and their running statistics remain agent-owned, and the generic runtime requires a GPU for the recursive `train()` entry point.
