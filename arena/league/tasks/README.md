# Arena tasks

This is the implementation package behind `arena.tasks`. It defines the public task vocabulary used by Arena’s manager, curriculum, training, and ADK adapter.

## Two layers

- [`framework/`](framework/README.md) contains the reusable semantics: tasks, difficulties, goals, criteria, rewards, JAX conformance gates, validation, and registration.
- [`games/`](games/README.md) contains the concrete examples shipped with this repository.

The framework is deliberately open-ended. A caller can author a new `Task`, use a `Goal` factory, compose a `TaskReward`, or start from [`games/TEMPLATE.py`](games/TEMPLATE.py). The manager does not special-case the shipped game names.

## Public lifecycle

```text
Goal / criterion + action-head coverage
                |
                v
Task + Difficulty + optional WorldSpec / Reward
                |
                v
validate() -> JAX goal/reward gate -> build_scene()
                |
                v
GameManager.register() -> build() -> evaluate()
```

Every task is validated before it is registered. The goal criterion must return one boolean per batch lane, and any named observation group must be readable from the current Arena observation contract. If a goal needs terrain or interaction evidence but the scene has no world provider, validation warns instead of pretending the zero-filled columns are meaningful.

## Current semantics worth knowing

- Goal success and transition reward are separate contracts.
- Target-health evidence is masked by `target_f32.visible`.
- `collect_item` and nearby-entity defeat are intentionally unavailable until their evidence is sound.
- `changed_terrain_at_interaction` detects a traversability change, not an exact block identity.
- Built-in world tasks use bounded, hashed WorldGen V2 providers; they do not generate terrain inside a rollout.

