

## 2026-10-25 — skill_agent: track's teacher caps its turn at the head's rate; holding skills' shares pooled over reachable ticks
command: the owner, 2026-10-25, "Cap track only" and "Pool over reachable ticks", on a teacher-only diagnosis of track's shortfall beside walk_to (track's TRAINING.md section 9).
result: The diagnosis was built by a Haiku subagent from a written spec. The wiring was done here.
- **The mechanism** (measured on the gen-7 render, teacher-only replays that reproduce the trainer's shares exactly):
  - the head is stored in world yaw;
  - the engine adds each yaw command to a commanded heading (`runtime/step.py` 167), and the head approaches it at 18 degrees a tick (`runtime/agent.py` 381-403);
  - the v2 rule measures each command from the observed head, so a 45-degree bin the head cannot follow leaves a lead that the next command adds to;
  - the lead pins the head at the ±45-degree window edge with the person 36 to 42 degrees inside it.
- **`neural_skills/track/teacher.py`:** `track_choices(..., step_degrees)` caps the yaw command at the largest codebook step within that rate (12.65).
- **`neural_skills/track/skill.py`:** `spec()` binds the capped teacher by default (`TEACHER_STEP_DEGREES` = 18). `spec(step_degrees=None)` is the rule of earlier runs.
- **Teacher share:**

| Lane kind | Before | Capped |
|---|---|---|
| alone | 0.957 | 0.989 |
| beside walk_to | 0.572 | 0.689 |

  These are lane means at seed 77001; seed 66001 agrees within a point. look_at keeps its rule: the cap removes its windup but changes its success by at most 1.6 points.
- **`training/skill_dagger.py` `_holding_share`:** a holding skill's share (track, follow) is pooled over the lanes' counted ticks, and the per-lane mean is kept as `lane_mean_share`. `counted_ticks` is new.
  - Beside walk_to, 42 of 128 lanes never have the person within reach and 34 have 1 to 10 counted ticks, yet each weighed as much as a long lane.
  - The pooled teacher share beside walk_to is 0.870 seen and 0.849 held out (capped).
  - The bars are unchanged.
- **Re-scored saved modules** (evaluation only, the same code for both trunks; TRAINING_PLAN 8.15): the people trunk's track module holds 0.943 alone and 0.648 beside walk_to. It was trained on the uncapped teacher; retraining on the capped one is next.
- **Tests (local):** `test_sa_track_teacher.py` 5 (the cap, the default binding, the pooled share); `test_sa_trunk_v3.py`'s teacher-success check now expects the pooled share. track, follow, skill_dagger_v3, trunk_v3 and bars: 47 passed.
gate: two passes, sharded, on ba67665 with track's capped teacher and the pooled holding share: pass 1 the step-9 baseline's 11 failing names, none new; the integration shard 7 passed; pass 2 1 failing name, a baseline one.
verdict: COMMITTED; not pushed.
