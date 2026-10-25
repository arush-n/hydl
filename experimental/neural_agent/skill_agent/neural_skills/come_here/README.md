# come_here — go to someone who calls

Walk to an entity and stop within reach of it: walk_to with the target taken from the entity's token instead of a
position. The request names no position; it points at entity 1 by its handle, and the trainer and the runtime resolve
the handle to the token row that holds the entity.

**Status (2026-10-24): built, not trained (stage 1 of `TRAINING.md`).** The plan is `TRAINING.md` (proposed, not
approved): what it does, its teacher, world, strategy, bars, stages and what it waits on. The build:

- `skill.py`: the request (no position; the entity handle; the gait and radius), the held-out region, the success test
  (planar distance to entity 1 within the radius), the metrics (walk_to's pace and radius), the spec, and the lowering.
  The verb `come_here` is appended to `../spec.py`'s `VERBS` (id 2).
- `task.py`: entity 1 placed 3 to 15 blocks from the spawn at any bearing (walk_to's range), at the floor's height; the
  held-out lanes place it at bearings 180 to 270 degrees from the spawn body.
- `teacher.py`: walk_to v3's rule (stop margin 0.25 blocks) aimed at entity 1's current position each tick.
- `train.py`: its strategy for `../../training/skill_dagger_v3.py --skill come_here`, with look_at beside it in half
  the lanes. It has no command line of its own (the v1 trainer does not place entity 1 per lane).

The certificate is `../../teaching/certificates.py --teacher come_here` (seen and held-out lanes). On seeds 77001,
66001, 66002 and 66003, 256 episodes each (2026-10-24), the teacher passes: T1 held success 1,024 / 1,024 seen and
held out; T2 median time ratio 1.013 seen, 1.119 held out (it turns first: the entity is behind); T3 pace on 98.7% and
100% of the steady lanes, the radius kept on every success, 0 illegal decisions. Its tests are
`tests/experimental/neural_agent/test_sa_come_here.py`. Not built here: `certify.sh`, `train.sh`, `evaluate.sh` (walk_to's
launchers) and an `__init__.py`. Its smoke and runs (2026-10-24, `../../docs/TRAINING_PLAN.md` 8.12): the module passes alone, not yet beside look_at. The index of
every skill is `../README.md`.
