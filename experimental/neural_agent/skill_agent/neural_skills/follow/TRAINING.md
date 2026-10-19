# follow: training plan

**Status (2026-10-19): proposed, not approved.** No stage runs before the owner approves it
(`../../docs/TRAINING_PLAN.md`, staged approval).

## 1. What it does

Follow a person: stay 2 to 5 blocks behind them while they move.

| | |
|---|---|
| Request | the person (the request's entity pointer), the band (near and far distance) |
| Success | the share of ticks inside the band, over the request's time |
| Done | none: a holding skill |
| Heads | locomotion and body yaw (walk_to's scope) |

## 2. Teacher

walk_to v3 (`../walk_to/teacher.py`) toward a point behind the person, idling while inside the band. New rule, so a
new certificate: T1 on the share in band, T2 against the person's own path length, T3 pace and legality.

## 3. World

A moving person, as for track (`../track/TRAINING.md` section 3: the simulator's target motion, or a scripted mover).
A person that approaches the agent cannot be followed, so the scripted mover is the likely need (inferred).

## 4. Student

The shared generalist with a new verb carrying the entity pointer and the band (scalar arguments), grown from the
release (A3) after come_here, which first teaches the entity argument on a static target.

## 5. Bars

- T5 on the share in band, at most 10 points under the teacher;
- held out: person speeds not trained, at least 0.9 of seen;
- no regression on walk_to, look_at and come_here.

## 6. Stages

1. **Build:** the scripted mover, the band metric, the teacher and its certificate. Gate: tests; T1 to T3.
2. **Smoke**, then **the run**, with a breakdown of misses by speed and turn rate of the person.

## 7. Waits on

come_here (the entity argument), a moving person in the task world, the owner's approval.
