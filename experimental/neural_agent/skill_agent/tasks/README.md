# skill_agent/tasks/ — the flat task world skills are trained and measured on

| Module | Lines | What it holds |
|---|---:|---|
| `simple_tasks.py` | 305 | The batched flat world with the NPC body (npc_walk) that the simple skills share: `make_flat_npc_environment` (legal observation on, the open-flat world, motion-timing profile 0 for 1/30 s ticks), the spawn and eye constants, and `run_episode(env, goals, success_fn, policy_fn, ...)`, which takes the skill's success test. Each skill's goals and test are in its folder: `../neural_skills/walk_to/task.py` (`WalkToTasks`) and `../neural_skills/look_at/task.py` (`LookAtTasks`). |

Gate: `test_sa_simple_tasks.py`. Planned, not built: the hard-case suites (combined requests and disturbances), where
a student must beat the scripted teachers (`../docs/TRAINING_PLAN.md` section 1b).
