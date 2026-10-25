# track — keep a moving thing in view

Hold the head on a moving entity (camera yaw and pitch: look_at's gaze scope). A holding skill: its success is the share
of ticks in view among the ticks the entity is within the head's reach, over the request's time. Its paths cross in front
of the agent; turning the body stays with walk_to or the decision layer, so track composes with walk_to.

**Status (2026-10-24): built, not trained.** The owner ruled the gaze-only form on 2026-10-24 (`TRAINING.md` section 8).
The trainer's holding-skill support and the mover in the rollout are built (`../../training/skill_dagger_v3.py`,
`../../training/bars.py`). The smoke and the run wait on the owner's approval (`TRAINING.md` stage 2). Follow (`../follow/`) is
built on the same trainer features.

| File | What it is |
|---|---|
| `skill.py` | the spec: the request (entity 1 by handle, look_at's tolerance scalar, the horizon as the time), the success test, `in_reach` (the share's counted ticks: mover_checks' ceiling rule) and the teacher binding; `lowering_batch` (2026-10-25, B5) is the numpy batch of its `lowering`, equal row by row |
| `task.py` | the goals: a person on a straight path crossing in front of the spawn at 3 to 10 blocks, walk or sneak (trained) or 2.0 to 2.3 blocks a second (held out); the horizon is the path's duration, capped at 300 ticks |
| `teacher.py` | look_at v2 aimed at entity 1 each tick, no lead, never done |
| `train.py` | its strategy: track alone, or with walk_to beside it in half the lanes |

Certificate: `python -m experimental.neural_agent.skill_agent.teaching.certificates --teacher track --seeds 77001,66001,66002,66003 --episodes 256 --out PATH --date DATE` (T1 share in view within reach at least 0.9, T3 legality; run speed for information).

Training (not run): `python -m experimental.neural_agent.skill_agent.training.skill_dagger_v3 --skill track --trunk TRUNK.npz --out REPORT.json --module-out MODULE.npz`.

Gates: `tests/experimental/neural_agent/test_sa_track.py`. The index of every skill is `../README.md`.
