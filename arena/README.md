# Arena

Arena is the task, world, evaluation, imitation, and training layer built on top of `hytalegym.jax`. It gives an agent a reproducible way to answer three questions:

1. What should the policy do?
2. Which observations and action heads are allowed to define that behaviour?
3. What evidence is strong enough to call the behaviour successful?

Arena is deliberately agent-neutral. It supplies scenes, task contracts, JAX rollout helpers, reward terms, imitation data structures, self-play primitives, and promotion gates; it does not contain a single “Arena agent.” The dependency boundary points one way: `adk` may adapt Arena, but Arena does not import ADK.

## Reading order

- [Task framework](league/tasks/framework/README.md) — goals, criteria, rewards, validation, and registries.
- [Shipped games](league/tasks/games/README.md) — concrete combat, movement, and WorldGen tasks.
- [Worlds](worlds.py) and [publication worlds](publication_worlds.py) — how bounded native worlds become JAX providers.
- [Training](training/README.md) — PPO, skill contracts, imitation, rewards, runtime orchestration, and self-play.
- [Imitation](imitation/README.md) — the algorithm-neutral native trace boundary.
- [Evaluation](evaluation/README.md) — held-out, replicated promotion evidence.

The README files in this tree are the public orientation layer. They describe the
interfaces a reader needs to use Arena; implementation details remain in the
source and package boundaries below.

## How a run moves through Arena

```text
Task / Difficulty / Goal / RewardConfig
                |
                v
        validation + JAX gate
                |
                v
 WorldSpec or PublishedWorldSpec
                |
                v
            ArenaScene
                |
                v
       ArenaHandle / GameManager
                |
                v
     lax.scan rollout + score
                |
                +--> PPO, imitation, self-play, evolution, or external agent
                |
                v
       replicated held-out evaluation
```

The environment and scoring path are kept separate. A goal answers “did the task succeed?” A `TaskReward` answers “what should be paid on each transition?” A reward can contain potential shaping, state terms, events, completion bonuses, failures, and step costs, so reward values must not be mistaken for success rates.

## Folder map

| Folder | Responsibility |
| --- | --- |
| [`league/`](league/README.md) | Opponent pools plus the physical owner of the task tree. |
| [`tasks/`](tasks/README.md) | Compatibility import namespace for `arena.league.tasks`. |
| [`curriculum/`](curriculum/README.md) | Promotion ladders and content-addressed training strategies. |
| [`imitation/`](imitation/README.md) | Native trace contracts, projectors, corpus composition, and capture. |
| [`training/`](training/README.md) | Learners, reward streams, skill stages, runtime, imitation training, and self-play. |
| [`evaluation/`](evaluation/README.md) | Multi-seed, held-out promotion assessment. |

The public package surface is the source and orientation documentation shown
above; validation suites are maintained outside that surface.

## Root modules

| File | What it owns |
| --- | --- |
| [`component.py`](component.py) | A parameter space plus a builder, producing validated labelled variants. |
| [`params.py`](params.py) | Typed scalar/choice domains, validation, sampling, and deterministic grids. |
| [`jax_contract.py`](jax_contract.py) | Named observation groups, readable-group policy, action-head spans, and loadout names. |
| [`jax_env.py`](jax_env.py) | `ArenaScene`, `ParameterizedPolicy`, JAX collection, and compiled evaluators. |
| [`manager.py`](manager.py) | Task registration, scene/evaluator caches, rollout recording, and `GameResult`. |
| [`worlds.py`](worlds.py) | Bounded built-in WorldGen V2 pools and world-provider binding. |
| [`publication_worlds.py`](publication_worlds.py) | Verified arbitrary publication pages with a resident capacity of one to four worlds. |
| [`runtime.py`](runtime.py) | Runtime-environment configuration re-exported from HytaleGym. |

## Important boundaries

- Observation and action widths come from the live HytaleGym contract. Older prose and smoke receipts contain historical widths; callers should use the imported contract constants.
- Hidden target fields are not evidence. Target-health goals and rewards mask them by visibility.
- Inventory tokens are named in the schema but are not currently readable by Arena, so `collect_item` is unavailable.
- Entity-slot disappearance is not death evidence, so nearby-entity defeat is unavailable.
- A terrain traversability change is not proof of a particular block placement.
- `arena.training.contracts` defines what a lesson rewards; `arena.training.runtime.contracts` defines how a training run is recorded and admitted. They are separate APIs with similar names.
