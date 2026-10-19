# look_at — turn the head to face a point and hold it there

The second neural skill. It trains inside the shared generalist (`../../model/generalist.py`) together with walk_to,
by DAgger distillation from a scripted teacher (`../../training/distill.py`, README 2.16). This folder holds what is
specific to look_at; the network, the trainer and the certificate harness are shared (`../README.md`).

| | |
|---|---|
| Request | slot 1 of the generalist's requests: the target as a body-frame position over 16 blocks and the tolerance; with `--look-direction`, also the unit direction from the eye to the target, in the head's frame with `--look-frame head` |
| Success | the head's yaw and pitch are both within 3 degrees of facing the target (`task.py`, `LookAtTasks.success`) |
| Done | the teacher's: the actual head within tolerance of where it aims |
| Heads | camera yaw (9 bins, up to 45 degrees a tick) and pitch (5 bins: -45, -1, 0, 1, 45 degrees): the `gaze` scope |
| Episodes | targets 2 to 10 blocks from the eye, within 40 degrees of the body yaw and 30 of pitch; horizon 90 ticks. While walking: within 30 degrees of the walk bearing and 20 of pitch |
| Held out | look-only targets 25 to 40 degrees left of the body: the left edge of the range, so it asks for extrapolation |
| World | flat, with the `npc_walk` body; the head turns within 45 degrees of the body |

## Status (2026-10-19): every look_at bar passes in run C'

Held success (on target 5 ticks in a row) on evaluation seed 77001, 256 lanes, about 85 per mode:

| Run (README 2.16) | Change | look only, seen / held out | T5 gap | held out / seen | walk while looking, seen (T5 gap) |
|---|---|---|---|---|---|
| A2 | the first run, teacher v2 | 0.776 / 0.459 | 22.4 points | 0.59 | 0.565 (34.1) |
| W | walk_to v3 only | 0.800 / 0.624 | 20.0 | 0.78 | 0.741 (16.5) |
| L | the look direction, body frame | 0.906 / 0.576 | 9.4 | 0.64 | 0.753 (15.3) |
| C | L and walk_to v3 | 0.847 / 0.576 | 15.3 | 0.68 | 0.847 (5.9) |
| C' | C, the direction in the head's frame | 0.988 / 1.000 | 1.2 | 1.01 | 0.882 (2.4) |

The bars are T5 at most 10 points and held out at least 0.90 of seen (`../../training/bars.py`).

What decided it, in two steps:
- **The direction (L).** A2's missed looks ended a median 5.2 degrees off against the 3-degree tolerance, worst for
  near targets, because the request carried only a position over 16 blocks: a 3-degree error at 2 blocks is about
  0.007 of an input. With the unit direction, near targets went from 0.55 to 0.89 (fresh episodes).
- **The head's frame (C').** A body-frame direction keeps its starting value while the head turns, so a held-out
  target stays outside the trained range for the whole look, and the student guesses: from the same input, L
  overshot in the held-out band (35 of 62 misses past the target) and C stopped short (61 of 82). In the head's frame
  the direction moves toward forward as the look closes, back into the trained range. On 768 fresh lanes per region
  the far band [32.5, 40] went from 0.56 (C) to 0.99, and the trained band next to it from 0.86 to 0.98.

Still open: in 3 of 512 fresh look-only lanes the head never pitches. The targets are 21.9 to 23.7 degrees up, and the
end error equals the target's pitch. They sit within a degree of 23, where the teacher's pitch choice switches from 1
degree a tick to 45 (the nearest of the 5 pitch bins), so this is inferred to be another decision on a boundary, like
walk_to's stop at the radius; not checked.

## Teacher versions

| Version | What changed | Certificate (private, `agent_skills/training_runs/skill_agent_a2/certificates/`) |
|---|---|---|
| v1 | steers from the commanded head yaw and pitch | `look_at_v1.json`: T1 to T3 pass; T5 fails: 28 % of its yaw labels depend on the commanded head, which the student cannot see, and its student completed no look (README 2.11) |
| v2 (registered) | the same rule on the observed state, where the head actually points | `look_at_v2.json`: T1 1,024 / 1,024, median 13 ticks, 0 illegal |

## Files

| File | What it is |
|---|---|
| `teacher.py` | `look_at_choices`: aim the head at the target within the 45-degree window around the body |
| `task.py` | `LookAtGoals`, `LookAtTasks` (`sample`, `success`) and the episode ranges |
| `certify.sh` | the teacher certificate, T1 to T3, on the four certificate seeds |
| `train.sh` | the generalist's current stage, which trains walk_to and look_at together |
| `evaluate.sh` | a run's look_at bars: T5 and held out for look only and both |
| `TRAINING.md` | the plan: what is done, what is next |

## Run

```sh
experimental/neural_agent/skill_agent/neural_skills/look_at/certify.sh            # -> certificates/look_at_v2.json
experimental/neural_agent/skill_agent/neural_skills/look_at/train.sh NAME         # -> skill_agent_a2/NAME/
experimental/neural_agent/skill_agent/neural_skills/look_at/evaluate.sh NAME      # the bars of that run
```

Every output goes to `agent_skills/training_runs/` (private, never staged); `train.sh` and `certify.sh` refuse to
overwrite. Training needs the owner's approval of its stage (`../../docs/TRAINING_PLAN.md`). Gates:
`test_sa_teachers.py`, `test_sa_simple_tasks.py`, `test_sa_certificates.py`, `test_sa_distill.py`, `test_sa_bars.py`.
