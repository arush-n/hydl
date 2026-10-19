# go_around: training plan

**Status (2026-10-19): proposed, not approved.** No stage runs before the owner approves it
(`../../docs/TRAINING_PLAN.md`, staged approval; it is run C4 there, after step_and_hop).

## 1. What it does

Reach a goal behind a wall: the third lesson of the navigation chain (owner, 2026-10-17).

| | |
|---|---|
| Request | walk_to's: the target position, the radius, the gait |
| Success and done | arrived: within the radius of the goal |
| Heads | walk_to's locomotion and body yaw, plus jump (from step_and_hop) |

## 2. Teacher

A planner: A* over the traversal graph (`hytalegym/jax/world/navigation/_astar.py`, not wired to any policy) plus
walk_to v3 toward the next waypoint. T2 is measured against the path tables' shortest-path time at the pace asked.
The teacher sees the whole map and the student does not, so T5 is where the imitation gap shows (TRAINING_PLAN 2c): if
the student cannot infer the detour from local geometry, the request gets a route hint (the next 2 or 3 waypoints).

## 3. World

The Region library on 0.7.2: recaptured (`41981ca`, 500 of 500 sections). The library swap and the product rebuild
(path tables, task zoo, catch-time horizons; `tools/worldgen/README.md` steps 1 to 5) are the owner's. Wall fixtures on
flat ground would let the first stage start earlier.

## 4. Student

Geometry tokens in the adapter and the trainer, as for step_and_hop (`../step_and_hop/TRAINING.md` section 4; neither
exists today). Its own module on the shared frozen trunk (`../../model/skill_module.py`), trained by its own strategy
(a `train.py` here, through `../../training/skill_dagger.py`; owner, 2026-10-19). Its name is appended to
`../spec.py`'s `VERBS` for its verb id. It starts from step_and_hop's module weights (the chain's third lesson). Its
strategy is a success-gated curriculum: the ratio of path length to straight-line distance grows only when development
success on the current level passes 70 % (TRAINING_PLAN 2b, rule 3).

## 5. Bars

- T5: at most 10 points under the teacher;
- held out: unseen worlds or wall layouts, at least 0.9 of seen;
- T2-style efficiency: the student's path time against the teacher's;
- the modules of walk_to, step_and_hop and look_at do not change (structural).

## 6. Stages

1. **Build:** the go_around task (goals behind walls, from the path tables), the planner teacher and its certificate.
2. **Smoke**, then **the curriculum run**, with a breakdown of misses by path ratio.

## 7. Waits on

The owner's library swap and the 0.7.2 path tables, geometry tokens in training, step_and_hop, the owner's approval.
