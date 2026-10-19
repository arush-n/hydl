# track: training plan

**Status (2026-10-19): proposed, not approved.** No stage runs before the owner approves it
(`../../docs/TRAINING_PLAN.md`, staged approval).

## 1. What it does

Keep a moving thing in view.

| | |
|---|---|
| Request | the entity (the request's entity pointer), the angle tolerance, the time |
| Success | the share of ticks with the head within tolerance of the entity, over the request's time |
| Done | none: a holding skill; the request's time limit ends it, not success |
| Heads | camera yaw and pitch (look_at's gaze scope) |

## 2. Teacher

look_at v2 (`../look_at/teacher.py`) aimed at the entity's position each tick. The head trails a command by several
ticks (README 2.11), so a moving target may need a lead (its position a few ticks ahead). T5 decides: lead only if the
student, trained without it, lags. Certificate: T1 on the share in view, T3 legal actions.

## 3. World: a moving entity

The flat task world's entity 1 is inert today (`../../tasks/simple_tasks.py`, `target_active=False`). Two options:
- the simulator's own target motion (entity 1 with `target_active`, `opponents/legacy/motion.py`,
  `_tick_target_motion`). **Inferred from its names, not checked:** it moves toward the agent, as an opponent;
- a scripted mover for entity 1 (a path or a random walk at a chosen speed), to build.

## 4. Student

The shared generalist with a new verb whose request sets the entity pointer, grown from the release by continued
distillation with rehearsal (A3), or a joint run.

## 5. Bars

- T5 on the share in view, at most 10 points under the teacher;
- held out: target speeds or paths not trained, at least 0.9 of seen;
- composition: track while walk_to runs, against the two alone (the "both" lanes of A2);
- no regression on walk_to and look_at.

## 6. Stages

1. **Build:** the moving entity (check option 1 first), the sampler, the teacher wrapper, the share-in-view metric and
   its certificate task. Gate: tests; the teacher's T1.
2. **Smoke**, then **the run** at A2's size, with a breakdown of misses by target speed and bearing.

## 7. Waits on

A moving entity in the task world; look_at's release (its held-out look must pass first, README 2.16); the owner's
approval.
