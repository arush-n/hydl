# look_at: training plan

**Status (2026-10-19): look_at's own training strategy (`train.py`) is built and unit-tested; its module is not
trained, as training is paused (owner, 2026-10-19).** In the joint reference network of runs A2 to C' (one head for
walk_to and look_at, README 2.16) every look_at bar passes. The shared method, the hypotheses and the staged approvals
are in `../../docs/TRAINING_PLAN.md`.

## 1. What is trained

look_at's own module (`../../model/skill_module.py`): the request-reading head of the generalist with look_at's own
weights, on the shared frozen trunk. Its strategy (`train.py`) is DAgger from teacher v2 (`teacher.py`), with the
request's direction in the head's frame, with walk_to beside it in half the lanes (by its own teacher, v3), the left
band held out, at A2's sizes, through the shared blocks (`../../training/skill_dagger.py`). The module reads walk_to's
request when both run and drives only look_at's heads; training it never changes the trunk or another skill's module.
Later, where the teacher is imperfect, KL-anchored RL from the distilled start (A4,
`../../docs/ARCHITECTURE_RESEARCH.md`).

## 2. The teacher's certificate (`../../docs/TRAINING_PLAN.md` 2c)

| Check | v1 | v2 (registered, used) |
|---|---|---|
| T1 success, 4 seeds x 256 episodes | 1,024 / 1,024 | 1,024 / 1,024 |
| T2 efficiency | median 13 ticks | median 13 ticks |
| T3 obedience: legal actions | 0 illegal | 0 illegal |
| T4 live replay on the 0.7.2 server | not run | not run: `../../teaching/certify_live.py` replays walk_to only |
| T5 imitability, through a student | fails: 28 % of its yaw labels read the commanded head, and its student completed no look | fails in A2 (22.4 points); passes in C' (1.2) |
| T6 recovery from student states | not run | not run |
| T7 inspection (failure classes) | T7-lite | T7-lite |

## 3. Stages

| Stage | What | Status |
|---|---|---|
| A2 (joint reference) | walk_to and look_at distilled together into one network, 6 iterations x 128 lanes x 2,000 steps | **passes with C'** (README 2.16) |
| Own module | look_at's module by its strategy (`train.py`) on trunk v1 (the encoder of run C'), A2's sizes; compared with the reference | strategy built and tested (2026-10-19); waits for training to reopen |
| Seeds | the module at two more training seeds: one seed decides nothing under about 15 points (TRAINING_NEEDS 16) | after the first module run |
| T4 | look_at's episodes replayed live: the head's lag behind a command is where the simulator and the server could differ | next: a look task in `certify_live.py` |
| steep pitch | the 3 of 512 fresh looks that end about 22 degrees off in pitch, on steep upward targets | open: a breakdown of the pitch head's choices first |
| Release 1 | the first versioned release of the skill modules and their trunk, with regression tests (owner, 2026-10-18) | after the seeds and T4 |
| H10, hard cases | beat the heuristic on combined requests, disturbances and unseen worlds (TRAINING_PLAN 1b) | designed |
| E5, E6, E12 | re-goal, calibrated status, legal sight with occlusion (TRAINING_PLAN C6 to C8) | planned |
| entity targets | look_at an entity, and track a moving one | `../track/` |

## 4. Bars (`../../training/bars.py`; README 2.16)

On held success (on target 5 ticks in a row, the first of them inside the horizon), for look only and both:
- T5: the teacher's rate less the student's at most 10 points;
- held out: the left-edge targets at least 0.9 of the seen rate;
- composition, reported: the both rate against walk x look in the same lanes.

The evaluation has about 85 lanes per mode, so each rate is good to about 8 points. L passed look-only T5 at 9.4
points and C failed it at 15.3, with the same look input. Re-scored on 256 fresh lanes per mode (seed 91001), the two
are alike and both fail, L by 12.1 points and C by 14.1, while C' passes by 1.6: L's pass was a draw near the edge.
A bar decision near its edge is re-scored on more lanes before it is trusted.

## 5. How to run each stage

`certify.sh` (T1 to T3); `train.sh NAME TRUNK.npz` (look_at's strategy, `train.py`, into
`agent_skills/training_runs/skill_modules/look_at/NAME/`); `evaluate.sh NAME` (the bars). A stage runs only once the
owner approves it; a smoke (`--iterations 2 --lanes 32 --steps 200`) comes before any longer run, and each run gets a
breakdown of how its misses end before the next change.
