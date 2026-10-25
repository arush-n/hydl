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

## 9. The gaze teacher's shortfall beside walk_to: diagnosis and an opt-in fix (2026-10-25, no training)

The shares below come from a teacher-only replay on the gen-7 render, seen lanes, seed 77001 and 66001, 256 lanes. The
replay mirrors `training/skill_dagger_v3.rollout`'s teacher-only path (the same keys, env steps, `skill_dagger.teach`
and `merge`, `in_reach`, `success`); it reproduces 8.15's teacher shares to their rounding (alone 0.9572, beside walk_to 0.5715, against 0.957 and 0.572).
Scripts and outputs are in the session scratchpad (`tt_logs/`), not the repo. Claims are marked measured or inferred.

**Where the head's yaw is stored (measured, and in the code).** The head is stored in world yaw, not relative to the body.
`types.py` 428-434 (`agent_head_yaw` and `desired_yaw` are the head's; `desired_body_yaw` is the body's); `runtime/step.py`
167 (`desired_yaw` += the yaw delta); `runtime/agent.py` 381-403 (`_head_steering`: the head approaches the commanded
heading at its rate, then is clipped to the body's window). On 29,545 unclamped, uncapped steps the head moves by exactly
(commanded minus actual) plus the executed delta, residual 0.0, including 472 steps with a body turn over 0.5 degrees. So
a body turn moves the gaze only where the window clamp binds (5,682 steps with the head on the window edge on a
body-turn step).

**The mechanism (measured).** The engine integrates the command: the stored commanded heading moves by exactly the
executed delta on every step (max error 1.5e-5 over 122,624 steps). The v2 teacher measures each command from the
observed head. A command the head cannot follow in one tick (18 degrees a tick; the codebook's top bin is 45) leaves the
commanded heading ahead of the head, and the next command, again measured from the observed head, adds to that lead
("windup"). At lane start, alone, the commanded heading ran a median 22 degrees from the bearing (max over the first 6 steps;
lane 0: commanded 150.2, bearing 128.2). The head overshot to the window's edge with the person 36.7 degrees inside it. In steady
state the same lead can pin the head on the edge while the person is inside (lane 140 at tick 6: head -45, person -41.7,
commanded lead -8.1).

**Classes (Step 1, per counted-but-not-in-view tick, exclusive, in priority b, c, a, d, e).** Counted = inside the
horizon and inside the reach; the replay's recomputed in-view and reach flags match the trainer's on every tick.

| Lane kind (seed 77001) | Counted ticks | Out of view | a lag after a body turn | b head saturated | c pitch | d turn limit | e other | Start (before the first in-view tick) | Windup (lead over 3) |
|---|---|---|---|---|---|---|---|---|---|
| alone | 21,535 | 779 | 0 | 511 | 32 | 131 | 105 | 406 | 559 |
| with walk_to | 3,209 | 595 | 268 | 167 | 63 | 11 | 86 | 389 | 374 |

- **Start-up dominates.** Of the out-of-view ticks, 52% (alone) and 65% (beside) come before a lane's first in-view tick.
  Beside, 264 of the 268 lag ticks are there: walk_to's first body turn (18 degrees a tick) coincides with the head's
  acquisition.
- **Steady-state misses** (alone 373, beside 206) are mostly the b class (the head on the edge with the person inside it,
  windup present). Class a is 4 steady ticks in total, so the spec's feed-forward for body turns is not the steady-state
  cause.
- **b is not a scoring question** (the spec expected a reach boundary question): its examples put the person 36 to 42
  degrees inside a head pinned at 45, well within the 48-degree reach. It is the teacher's windup.
- **The reach band** (the person beyond 45 degrees but inside the 48-degree reach) accounts for 68 alone ticks only.

**Scoring, not the teacher (measured).** The beside share is a mean over lanes with at least one counted tick. 42 of 128
beside lanes never have the person within reach: the person's yaw relative to the body has a median minimum of 55
degrees over the horizon, because walk_to turns the body a median 241 degrees. A further 34 lanes have 1 to 10 counted
ticks, and they score 0.27 on average (0.47 with the fix below). Pooled over ticks, the beside share is 0.815 (observed)
and 0.870 (fixed), against the lane mean's 0.572 and 0.689. Which lanes count is the owner's call (`TRAINING.md` 5's
bar is on this share).

**The fix (opt-in, default off): `step_degrees`.** `track_choices(..., step_degrees=18.0)` caps the yaw command at the
largest codebook step within the head's per-tick rate (12.65 degrees on this codebook). The default is the registered
rule, which a test pins (`test_sa_track_teacher.py`). The rate is the head's `agent_max_head_rotation_degrees` x
`agent_steering_relative_turn_speed` / ticks per second, 360 x 1.5 / 30 = 18 (measured from the gen-7 render's params);
a caller passes it, and `spec()` does not wire it, so no training run uses it yet.

| Lane kind | Seed | Default (observed) | Step cap 18 (kept) | Commanded reference (tested, not kept) | Both |
|---|---|---|---|---|---|
| alone | 77001 | 0.957 | 0.989 | 0.990 | 0.989 |
| with walk_to | 77001 | 0.572 | 0.689 | 0.689 | 0.668 |
| alone | 66001 | 0.957 | 0.987 | 0.989 | 0.987 |
| with walk_to | 66001 | 0.575 | 0.682 | 0.679 | 0.664 |

The numbers are the trainer's share (the mean over lanes with counted ticks), teacher only. The pooled beside share moves
0.815 to 0.870 (77001) and 0.822 to 0.876 (66001). After the step cap (seed 77001) the alone misses are 239 ticks, 232 of
them at start; the beside misses are 417, 252 at start and 165 steady (b 58, c 38, e 58, a 11).

- **The step cap** is the kept option: it leaves the labels on the observed state, as v2 has them (the student's inputs;
  the learnability reasoning is in `teaching/teachers.py`). Inferred, not measured: that a student learns these labels as
  well as today's.
- **The commanded reference** (the labels measured from the commanded heads, `runtime/step.py` 167) gives the same gain
  within one point, and is not kept: its labels depend on the commanded heading, which the student does not see (the
  `teachers.py` docstring records 28% unpredictable v1 labels). Owner's call if wanted.
- **Both together** are worse on beside walk_to (0.668 and 0.664): the cap slows the correction of a lead that the
  reference change would otherwise remove.

**Not done (and why).** The feed-forward for a body turn (the spec's (a) fix) is not built: the steady-state lag after a
turn is 4 ticks, and the start-phase overlap is acquisition. No training; no change to `skill.py`, `train.py`, look_at's
teacher or walk_to's. Owner questions: which lanes count for the share (section above), and whether the step cap is
wired into track's training (a run would need the owner's approval, `TRAINING.md` stage 2).

**The owner's rulings (2026-10-25).**
- "Cap track only": `skill.py`'s `spec` now binds the capped teacher by default (`TEACHER_STEP_DEGREES` = 18).
  `spec(step_degrees=None)` is the uncapped rule that 8.12's and 8.15's runs used. look_at keeps its registered teacher:
  the cap removes its windup but changes its success by at most 1.6 points (`tt_logs/look_at/`).
- "Pool over reachable ticks": a holding skill's share (`training/skill_dagger.py` `_holding_share`, so track's and
  follow's) is now the counted ticks in view over the counted ticks, pooled over the lanes. The per-lane mean of earlier
  reports is kept beside it as `lane_mean_share`. The bars are unchanged.
