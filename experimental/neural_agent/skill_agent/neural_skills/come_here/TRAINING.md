# come_here: training plan

**Status (2026-10-19): proposed, not approved.** No stage runs before the owner approves it
(`../../docs/TRAINING_PLAN.md`, staged approval). This is the next skill that needs no new simulator mechanic.

## 1. What it does

Go to someone who calls: walk to an entity and stop within reach of it.

| | |
|---|---|
| Request | the entity (the request's entity pointer, not a position), the arrival radius, the gait |
| Success and done | the planar distance to the entity is within the radius |
| Heads | locomotion and body yaw (walk_to's scope) |
| Difference from walk_to | the target comes from the entity's token, so it can move; the request has no position |

## 2. Why it is next

The flat task world already holds a second entity, placed per lane at reset (`../../tasks/simple_tasks.py`,
`make_flat_npc_environment`: `reset` takes `other_xyz`). A static entity at a sampled place is enough for the first
stage; a moving one comes with track and follow.

## 3. Teacher

walk_to v3 (`../walk_to/teacher.py`) aimed at the entity's position each tick: the certified rule with a new target
source. Certificate: T1 to T3 as walk_to's (`../../teaching/certificates.py` gains a come_here task); T5 through its
student.

## 4. Student

The shared generalist (`../../model/generalist.py`) with a new verb whose request sets the entity pointer
(`GeneralistRequests.entity_pointer`). Two ways, A3's choice (`../../docs/ARCHITECTURE_RESEARCH.md`): continued
distillation from the walk_to and look_at release with rehearsal of both, or a joint run from scratch at A2's size.

## 5. Bars (as walk_to's, `../../training/bars.py`)

- T5: teacher minus student at most 10 points, on held success;
- held out: entity bearings 180 to 270 degrees from the spawn body (walk_to's region) at least 0.9 of seen;
- E4: pace within 5 % and the radius kept, in 95 %;
- no regression: walk_to and look_at stay within tolerance of their bars (A3).

## 6. Stages

1. **Build (no training):** the sampler places the entity; the teacher wrapper; the certificate task; the verb and its
   request in `../../training/distill.py`. Gate: unit tests, and the certificate passes T1 to T3.
2. **Smoke** at 2 iterations x 32 lanes x 200 steps, end to end, before any long run.
3. **The run** at A2's size, scored on the bars, with a breakdown of how misses end.

## 7. Waits on

The walk_to and look_at release (the generalist A3 grows from), and the owner's approval of stage 1.

## 8. If it fails

A T5 failure with the entity's token in view points to the pointer: check that the student attends to the pointed
token (an ablation with the pointer cleared) before changing the teacher.
