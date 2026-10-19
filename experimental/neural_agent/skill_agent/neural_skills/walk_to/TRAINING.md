# walk_to: training plan

**Status (2026-10-19): walk_to's own training strategy (`train.py`) is built and unit-tested; its module is not
trained, as training is paused (owner, 2026-10-19).** In the joint reference network of runs A2 to C' (one head for
walk_to and look_at, README 2.16) every walk_to bar passes. The shared method, the hypotheses and the staged approvals
are in `../../docs/TRAINING_PLAN.md`.

## 1. What is trained

walk_to's own module (`../../model/skill_module.py`): the request-reading head of the generalist with walk_to's own
weights, on the shared frozen trunk. Its strategy (`train.py`) is DAgger from teacher v3 (`teacher.py`, the stop
margin), with look_at beside it in half the lanes (by its own teacher), rear-left targets held out, at A2's sizes,
through the shared blocks (`../../training/skill_dagger.py`). The module reads look_at's request when both run and
drives only walk_to's heads; training it never changes the trunk or another skill's module. Later, where the teacher
is imperfect, KL-anchored RL from the distilled start (A4, `../../docs/ARCHITECTURE_RESEARCH.md`).

## 2. The teacher's certificate (`../../docs/TRAINING_PLAN.md` 2c)

| Check | v1 | v2 (registered) | v3 (used) |
|---|---|---|---|
| T1 success, 4 seeds x 256 episodes | 1,024 / 1,024 | 1,024 / 1,024 | 1,024 / 1,024 |
| T2 efficiency, median against the kinematic minimum | 1.0004 | 1.037 | 1.037 |
| T3 obedience: pace and legal actions | pass | pass | pace on all 1,015 steady lanes, 0 illegal of 207,561 |
| T4 live replay on the 0.7.2 server | 24 / 24, as simulated | not run | not run: `certify_live.py` has no margin option |
| T5 imitability, through a student | not measured at size | fails in A2 by 16.3 points | passes: 0 points in W, C and C' |
| T6 recovery from student states | not run | not run | not run |
| T7 inspection (failure classes) | T7-lite | T7-lite | T7-lite: no failure class |

## 3. Stages

| Stage | What | Status |
|---|---|---|
| A2 (joint reference) | walk_to and look_at distilled together into one network, 6 iterations x 128 lanes x 2,000 steps | **passes with C'** (README 2.16) |
| Own module | walk_to's module by its strategy (`train.py`) on trunk v1 (the encoder of run C'), A2's sizes; compared with the reference | strategy built and tested (2026-10-19); waits for training to reopen |
| Seeds | the module at two more training seeds: one seed decides nothing under about 15 points (TRAINING_NEEDS 16) | after the first module run |
| T4 for v3 | v3's episodes replayed live, then v3 registered | next: add the margin to `certify_live.py` |
| Release 1 | the first versioned release of the skill modules and their trunk, with regression tests (owner, 2026-10-18) | after the seeds and T4 |
| H10, hard cases | beat the heuristic on combined requests, disturbances and unseen worlds (TRAINING_PLAN 1b) | designed |
| E5, re-goal | a target changed mid-episode is recovered (TRAINING_PLAN C6) | planned |
| E6, status | done, stuck and P(success) calibrated (C7) | planned |
| E12, legal sight | success at FOV16 and FOV32 with occlusion (C8) | planned |
| sprint | sprint_to: the teacher sprints forward only, and the sampler draws walk, run and sneak | open |
| terrain | the navigation chain's next lessons | `../step_and_hop/`, `../go_around/` |

## 4. Bars (`../../training/bars.py`; README 2.16)

On held success (on target 5 ticks in a row, the first of them inside the horizon), for walk only and both:
- T5: the teacher's rate less the student's at most 10 points;
- held out: the rear-left bearings at least 0.9 of the seen rate;
- E4: pace within 5 % on at least 95 % of the steady lanes, and the radius kept on at least 95 % of the successes.

The evaluation has about 85 lanes per mode, so each rate is good to about 8 points: a bar decision near its edge is
re-scored on more lanes before it is trusted.

## 5. How to run each stage

`certify.sh` (T1 to T3); `train.sh NAME TRUNK.npz` (walk_to's strategy, `train.py`, into
`agent_skills/training_runs/skill_modules/walk_to/NAME/`); `evaluate.sh NAME` (the bars). A stage runs only once the
owner approves it; a smoke (`--iterations 2 --lanes 32 --steps 200`) comes before any longer run, and each run gets a
breakdown of how its misses end before the next change.
