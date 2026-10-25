# follow: training plan

**Status (2026-10-19): proposed, not approved.** No stage runs before the owner approves it
(`../../docs/TRAINING_PLAN.md`, staged approval). **Stage 1 built (2026-10-24, no training):** the teacher passes T1
to T3 (`README.md`); stages 2 and 3 ran (owner: "yes for them all"; `../../docs/TRAINING_PLAN.md` 8.12): the
module falls short of its teacher on the frozen V1 trunk.

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

Its own module on the shared frozen trunk (`../../model/skill_module.py`), trained by its own strategy (a `train.py`
here, through `../../training/skill_dagger.py`; owner, 2026-10-19). Its name is appended to `../spec.py`'s `VERBS` for
its verb id. Its request carries the entity pointer and the band (scalar arguments). It may start from come_here's
module weights, which first learns the entity argument on a static target; come_here's own module never changes.

## 5. Bars

- T5 on the share in band, at most 10 points under the teacher;
- held out: person speeds not trained, at least 0.9 of seen;
- the modules of walk_to, look_at and come_here do not change (structural).

## 6. Stages

1. **Build:** the scripted mover, the band metric, the teacher and its certificate. Gate: tests; T1 to T3.
2. **Smoke**, then **the run**, with a breakdown of misses by speed and turn rate of the person.

## 7. Waits on

come_here (the entity argument), a moving person in the task world, the owner's approval.

## 8. Measured: the world and the teacher (2026-10-24, no training)

The moving person is the scripted mover (`../track/TRAINING.md` section 8).

**The teacher check** (`../../bench/mover_checks.py`):
- the teacher: walk_to v3 toward a point 3.5 blocks behind the person, radius 1, idle inside its stop radius;
- the learner starts 3.5 blocks behind the person;
- 256 lanes for 300 ticks, on seeds 0 and 1.

The share of ticks 2 to 5 blocks from the person:

| Person's speed | Learner runs | Learner sprints |
|---|---|---|
| walk 1.5 | 0.961, 0.964 | 0.968, 0.970 |
| sneak 2.75 | 0.942, 0.936 | 0.949, 0.943 |
| run 5.0 | 0.548, 0.544 | 0.636, 0.644 |

- At walk and sneak speed the teacher holds the band 94 to 97% of the time.
- Behind a runner it holds the band 55% of the time at run and 64% at sprint. The sprint (6.4 blocks a second) is
  barely faster than the runner (5.0), and the person turns.

**RULED (owner, 2026-10-24): sprint; walk and sneak; runners parked.**
- The teacher sprints when behind.
- The first run trains on walking and sneaking people (1.5 and 2.75 blocks a second).
- A speed between them is held out, for the "person speeds not trained" bar.
- People at run speed wait for a teacher that keeps up (for example, one that cuts corners).

The rejected options: including runners now (T1 would fail on them), and a new gait-by-distance teacher (it needs its
own certificate first).
