# walk_to — go to a place at a chosen pace and stop inside a radius

A neural skill with its own module and its own training strategy (owner, 2026-10-19): `train.py` trains
walk_to's module, the request-reading head of `../../model/generalist.py` with walk_to's own weights, on the
shared frozen trunk, by DAgger from its scripted teacher (`../../training/skill_dagger.py`). The module reads
look_at's request when both run, since it trains beside it, and drives only walk_to's own heads. This folder
holds everything specific to walk_to (`../README.md`).

| | |
|---|---|
| Request | slot 0 of the generalist's requests: the target as a body-frame position over 16 blocks, the radius, the gait |
| Success | the planar distance to the target is within the radius (`task.py`, `WalkToTasks.success`) |
| Done | the same test (the teacher's done signal) |
| Heads | locomotion (idle, or a gait in one of 8 body-relative directions) and body yaw: the `locomotion` scope |
| Episodes | targets 3 to 15 blocks from the spawn, radius 0.5 to 1.5 blocks, gait walk, run or sneak; horizon: the time at the gait's speed x 1.5, plus 30 ticks |
| Held out | targets at bearings 180 to 270 degrees, clockwise from the spawn body's forward (the rear left) |
| World | flat, with the `npc_walk` body the game server's NPC uses (owner, 2026-10-18; `../../tasks/simple_tasks.py`) |

## Status (2026-10-19): its own module is not trained yet; in the joint reference network every walk_to bar passes

walk_to's strategy (`train.py`) is built and unit-tested, and its module waits for training to reopen (paused by the
owner, 2026-10-19). What is measured so far comes from the joint reference network of runs A2 to C' (one head for
walk_to and look_at together, README 2.16), which the module will be compared against. Run C'
(`--walk-stop-margin 0.25 --look-direction --look-frame head`) passes all of A2's bars, walk_to's and look_at's. Held
success (on target 5 ticks in a row) on evaluation seed 77001, 256 lanes, about 85 per mode:

| Run (README 2.16) | Change | walk only, seen / held out | T5 gap | walk lanes missed (of 342) | pace within 5 % | radius kept |
|---|---|---|---|---|---|---|
| A2 | the first run, teacher v2 | 0.837 / 0.837 | 16.3 points | 55 | 96.7 % | 100 % |
| W | teacher v3 (the stop margin) | 1.000 / 1.000 | 0 | 0 | 98.0 % | 100 % |
| L | the look direction (no margin) | 0.988 / 0.965 | 1.2 | 22 | 96.7 % | 100 % |
| C | v3 and the look direction | 1.000 / 1.000 | 0 | 1 | 97.3 % | 100 % |
| C' | C and the head frame | 1.000 / 1.000 | 0 | 2 | 98.0 % | 100 % |

What decided it: without the margin, every missed walk parks a few centimetres outside the radius and never moves
again, because the teacher stops exactly on the edge and the student acts by argmax on a state that no longer changes
(A2: 64 of 66 misses, a median 0.02 blocks short). With v3 the walk-only lanes miss nothing in W, C and C'; C and C'
miss 1 and 2 walks of 342, all in walk-while-looking lanes.

## Teacher versions

| Version | What changed | Certificate (private, `agent_skills/training_runs/skill_agent_a2/certificates/`) |
|---|---|---|
| v1 | steers from the commanded body yaw | `walk_to_v1.json`: T1 1,024 / 1,024, 1.0004 x the kinematic minimum; T4 live 24 / 24 |
| v2 (registered) | the same rule on the observed state, the actual body yaw (README 2.11) | `walk_to_v2.json`: T1 1,024 / 1,024, T2 median 1.037 |
| v3 | v2, stopping 0.25 blocks inside the radius, or half the radius if that is less (`WALK_TO_V3_STOP_MARGIN_BLOCKS`) | `walk_to_v3.json`: T1 1,024 / 1,024, T2 median 1.037, T3 pace on all 1,015 steady lanes, 0 illegal of 207,561 |

v3 is used through `--walk-stop-margin 0.25`; the registry still names v2 (`../../teaching/teachers.py`, `TEACHERS`),
and promoting v3 waits on its T4 live replay (`TRAINING.md`).

## Files

| File | What it is |
|---|---|
| `teacher.py` | `walk_to_choices` (v2 at margin 0, v3 above), `walk_to_version`, the v3 margin |
| `task.py` | `WalkToGoals`, `WalkToTasks` (`sample`, `success`) and the episode ranges |
| `skill.py` | walk_to as the training blocks see it (`spec()`): its request, its sampler and held-out region, its teacher binding |
| `train.py` | **its training strategy**: DAgger from v3, look_at beside it in half the lanes, rear-left targets held out, A2's sizes |
| `train.sh` | runs `train.py` into `agent_skills/training_runs/skill_modules/walk_to/NAME/` |
| `certify.sh` | the teacher certificate, T1 to T3, on the four certificate seeds (v3 by default) |
| `evaluate.sh` | a run's walk_to bars: T5 and held out for walk only and both, and E4 |
| `TRAINING.md` | the plan: what is done, what is next |

## Run

```sh
experimental/neural_agent/skill_agent/neural_skills/walk_to/certify.sh            # -> certificates/walk_to_v3.json
experimental/neural_agent/skill_agent/neural_skills/walk_to/train.sh NAME TRUNK.npz  # -> skill_modules/walk_to/NAME/
experimental/neural_agent/skill_agent/neural_skills/walk_to/evaluate.sh NAME      # the bars of that run
```

Every output goes to `agent_skills/training_runs/` (private, never staged); `train.sh` and `certify.sh` refuse to
overwrite. Training needs the owner's approval of its stage (`../../docs/TRAINING_PLAN.md`). Gates:
`test_sa_teachers.py`, `test_sa_simple_tasks.py`, `test_sa_certificates.py`, `test_sa_skill_modules.py`,
`test_sa_distill.py`, `test_sa_bars.py`. `TRUNK.npz` is the shared trunk (`../../model/skill_module.py`,
`save_trunk`); the first one is to be exported from the encoder of run C' (`trunk_from_run`).
