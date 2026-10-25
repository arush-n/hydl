# follow — stay 2 to 5 blocks behind a person who walks

Stay 2 to 5 blocks behind a person while they walk. A holding skill: its success is the learner's planar distance to
entity 1 inside the band, and its share is the ticks in the band over the horizon (300 ticks, 10 s). The teacher sprints
when behind (owner, 2026-10-24: sprint; walk and sneak; runners parked; `TRAINING.md` section 8).

**Status (2026-10-24): built, not trained (stage 1 of `TRAINING.md`).** The plan is `TRAINING.md` (proposed, not approved).
The build follows track's files, on the trainer's holding-skill support and its per-lane mover
(`../../training/skill_dagger_v3.py`, `../../training/bars.py`). The smoke and the run wait on the owner's approval.

| File | What it is |
|---|---|
| `skill.py` | the spec: the request (entity 1 by its handle, the band's near and far edges as the two scalars, no position), the success test, the lowering, and the teacher binding (holding; every tick counts); `lowering_batch` (2026-10-25, B5) is the numpy batch of the lowering, equal row by row |
| `task.py` | the person: a looped random walk of 8 waypoints within 8 blocks (`bench/mover_checks`' path), walking 1.5 or sneaking 2.75 blocks a second (held out: 2.0 to 2.3); each lane's path is translated so the learner spawns 3.5 blocks behind the person's start, facing it; 300 ticks |
| `teacher.py` | walk_to v3's rule toward the band's middle (3.5 blocks) behind the person, along their velocity (their yaw when standing), radius 1, sprint, idle inside the stop radius; never done |
| `train.py` | its strategy: follow alone, or with track beside it in half the lanes (track by its own teacher, so the module watches the person while it follows) |

The verb `follow` is appended to `../spec.py`'s `VERBS` (id 4).

**Certificate** (`python -m experimental.neural_agent.skill_agent.teaching.certificates --teacher follow --seeds 77001,66001,66002,66003 --episodes 256 --out PATH --date DATE`): the teacher alone, seen speeds and held-out speeds, 256 lanes per seed. On 2026-10-24 (`--out` not kept in the tree):
- **T1** share in band: seen 0.935 (walk 0.968 on 504 lanes, sneak 0.903 on 520 lanes), bar 0.9: PASS. Held out 0.945 (2.0 to 2.3, 1,024 lanes): PASS.
- **T2** learner's planar path over the person's, median over lanes: seen 1.174 (p90 1.364), held out 1.199 (p90 1.323), bar 1.3 (provisional): PASS.
- **T3** illegal decisions: 0 in both regions: PASS.
- Information, run speed 5.0 (not gated): share in band 0.583, path ratio median 0.947.
- Overall: PASS.

Measured beside the certificate (scratch, 256 lanes, seed 77001, one person path each): the built teacher gives walk 0.968
and sneak 0.909 (held out 2.15: 0.949). Driving it from the commanded yaw instead, as `bench/mover_checks` does, gives walk
0.968 and sneak 0.946 (held out 0.957), which is `run_follow`'s own number. The difference is the state the teacher reads:
the built teacher uses walk_to's registered observed state (`teaching/teachers.state_for`), as come_here and track do, so
its labels come from the body yaw the student sees. The owner or coordinator decides which convention follow keeps.

**Tests:** `tests/experimental/neural_agent/test_sa_follow.py` (sampler, learner start and facing, the person's velocity in
the state, the teacher on 32 lanes, the request, the verb, the holding skill's done, a rollout, the bars on a fixture report,
and an idle teacher failing T1).

**Not built here:** `certify.sh`, `train.sh`, `evaluate.sh` (walk_to's launchers) and an `__init__.py` (as track and
come_here). Training waits on the owner's approval (`../../docs/TRAINING_PLAN.md`). The index of every skill is `../README.md`.
