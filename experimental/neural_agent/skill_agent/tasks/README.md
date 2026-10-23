# skill_agent/tasks/ — the worlds skills are trained and measured on: the flat task world, the duel and the league

| Module | Lines | What it holds |
|---|---:|---|
| `simple_tasks.py` | 305 | The batched flat world with the NPC body (npc_walk) that the simple skills share: `make_flat_npc_environment` (legal observation on, the open-flat world, motion-timing profile 0 for 1/30 s ticks), the spawn and eye constants, and `run_episode(env, goals, success_fn, policy_fn, ...)`, which takes the skill's success test. Each skill's goals and test are in its folder: `../neural_skills/walk_to/task.py` (`WalkToTasks`) and `../neural_skills/look_at/task.py` (`LookAtTasks`). |
| `duel.py` | 191 | The duel world (Stage L, step L1) on the coordinator's two-actor arena (`experimental/neural_agent/tasks/duel_arena.py`: both bodies take factors, the built-in AI is off): `arena` (the open-flat sector and sensor range, both bodies in the Trork role by default (`ROLES`, the coordinator's `roles=`) at 61 maximum health), `mirrored` placements (symmetric under a half turn), `fight(duel, first, second, placement, ticks, seed, delays)` with controllers `(observation, mask, context)` and the ticks since each entity was last hurt, and L1's mirror check (CLI). |
| `league.py` | 224 | The league (Stage L, step L2): the decision layer's combat styles COUNTER, PRESSURE and HIT_AND_RUN as pure-JAX controllers over the bodies (named after `jev/combat_styles.py`'s programs), with the scripted priority as PRIORITY; a round robin of every ordered pair in both seats as one batch; Bradley-Terry ratings on the Elo scale; L2's cycle gate (CLI). |
| `archery.py` | 165 | The archery range (2026-10-20): the shortbow body (`../runtime/scripted_bodies.py`) on the duel with the coordinator's `arrows=` against a still, strafing (runs of 20 to 60 ticks each way) or approaching body, per draw tier and distance: arrows loosed, hits, damage and damage per second, for the decision layer's hold (CLI). The shooter's body turns to face the target as a locomotion body would (`--no-face`: the bow alone, whose head turns at most 45 degrees off the body). |

Gates: `test_sa_simple_tasks.py`, `test_sa_duel.py`, `test_sa_league.py`, `test_sa_archery.py`. Planned, not built: the hard-case suites (combined requests and disturbances), where
a student must beat the scripted teachers (`../docs/TRAINING_PLAN.md` section 1b).

Run L1's mirror check (one JSON line per weapon) and L2's round robin (one JSON line); on gen 7 through
`../runtime/gen7.py`:

```sh
JAX_PLATFORMS=cpu nice -n 10 .venv/bin/python -m experimental.neural_agent.skill_agent.tasks.duel --batch 64 \
    --ticks 600 --seed 0 --loadouts iron_sword,iron_battleaxe,iron_mace,iron_daggers [--delay-ticks 20]
JAX_PLATFORMS=cpu nice -n 10 .venv/bin/python -m experimental.neural_agent.skill_agent.tasks.league --lanes 32 \
    --ticks 600 --seed 0 --styles COUNTER,PRESSURE,HIT_AND_RUN,PRIORITY --loadout iron_sword
```
