# track: training plan

**Status (2026-10-19): proposed, not approved.** No stage runs before the owner approves it
(`../../docs/TRAINING_PLAN.md`, staged approval). **Stage 1 built (2026-10-24, no training):** the teacher passes T1
and T3 (`README.md`); stages 2 and 3 ran (owner: "yes for them all"; `../../docs/TRAINING_PLAN.md` 8.12): the
module falls short of its teacher on the frozen V1 trunk.

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
  `_tick_target_motion`). **Checked 2026-10-24, ruled out:** it is an opponent controller that chases the agent,
  holds a distance band and strafes (section 8);
- a scripted mover for entity 1 (a path or a random walk at a chosen speed), to build.

## 4. Student

Its own module on the shared frozen trunk (`../../model/skill_module.py`), trained by its own strategy (a `train.py`
here, through `../../training/skill_dagger.py`; owner, 2026-10-19). Its name is appended to `../spec.py`'s `VERBS` for
its verb id. Its request sets the entity pointer. It may start from look_at's module weights (the gaze is look_at's);
look_at's own module never changes. Its strategy trains it with walk_to beside it in some lanes, so it learns to track
while walking.

## 5. Bars

- T5 on the share in view, at most 10 points under the teacher;
- held out: target speeds or paths not trained, at least 0.9 of seen;
- composition: track while walk_to runs, against the two alone (the "both" lanes of A2);
- walk_to's and look_at's modules do not change (structural).

## 6. Stages

1. **Build:** the moving entity (check option 1 first), the sampler, the teacher wrapper, the share-in-view metric and
   its certificate task. Gate: tests; the teacher's T1.
2. **Smoke**, then **the run** at A2's size, with a breakdown of misses by target speed and bearing.

## 7. Waits on

A moving entity in the task world; look_at's release (its held-out look must pass first, README 2.16); the owner's
approval.

## 8. Measured: the world and the teacher (2026-10-24, no training)

**The world.** Option 2 is built: an opt-in scripted mover for entity 1 (`../../tasks/simple_tasks.py`, `Mover`), a
path per lane at a speed from the agent's own gait table (walk 1.5, sneak 2.75, run 5.0 blocks a second). With no mover
the world is bit-identical to before. The entity token carries the mover's position and velocity, but not its yaw.

**The teacher check** (`../../bench/mover_checks.py`). It runs 256 lanes for 300 ticks, with a looped random walk of 8
waypoints within 8 blocks of the spawn, on seeds 0 and 1. The table gives the share of ticks with the head within
look_at's 3-degree tolerance:

| Speed | No lead | 3-tick lead | Ceiling (the head's 45 degrees each side, body still) |
|---|---|---|---|
| walk 1.5 | 0.254, 0.254 | 0.248, 0.246 | 0.263, 0.262 |
| sneak 2.75 | 0.204, 0.207 | 0.184, 0.180 | 0.231, 0.236 |
| run 5.0 | 0.175, 0.192 | 0.121, 0.132 | 0.258, 0.269 |

- **The head's range binds.** With the body still, a person walking around the agent is within the head's reach only
  about a quarter of the time. Within reach, the teacher keeps them in view 97% (walk), 88% (sneak) and 68% (run) of
  the time (seed 0).
- **A lead hurts** at every speed, so the teacher uses none.
- **Reading lag:** about 3 of run's 8 lost points come from reading success one tick after the aim (0.207 when read
  before the step, against 0.175).

**RULED (owner, 2026-10-24): gaze only, scored within reach.** A gaze-only teacher cannot pass T1 over every tick on
this world, so:
- track stays on look_at's gaze heads (camera yaw and pitch);
- its success is the share of ticks in view among the ticks the person is within the head's reach;
- its paths cross in front of the agent.

Turning the body stays with walk_to or the decision layer, so track composes with walk_to (tracking while walking).
The rejected option: track turns the body at the head's limit. Body yaw would then join its heads, and it would compete
with walk_to's body yaw when both run.
