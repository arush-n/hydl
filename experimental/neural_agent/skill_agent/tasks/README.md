# skill_agent/tasks/ — the episodes skills are trained and measured on

| Module | Lines | What it holds |
|---|---:|---|
| `simple_tasks.py` | 448 | Batched flat-world episodes with the NPC body (npc_walk) for walk_to and look_at: `make_flat_npc_environment` (legal observation on, the open-flat world, motion-timing profile 0 for 1/30 s ticks), `WalkToTasks`, `LookAtTasks`, `run_episode`. |

Gate: `test_sa_simple_tasks.py`. Planned, not built: the hard-case suites (combined requests and disturbances), where
a student must beat the scripted teachers (`../docs/TRAINING_PLAN.md` section 1b).
