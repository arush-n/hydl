# Training runs

This directory contains concrete experiment runners rather than reusable algorithm primitives.

## `pursuit_run.py`

The pursuit runner trains a recurrent policy in a two-actor Region or authored flat arena. It can use:

- saved Region artifacts or a declared design world;
- scripted flee, adaptive flee, standoff, mixed, or frozen-source targets;
- a separate opponent checkpoint for recursive pursue/evade experiments;
- fixed common-random-number evaluation milestones;
- asynchronous read-only evaluation on a separate device when available;
- replay archives for selected lanes;
- KL and median log-ratio divergence guards;
- milestone, latest, and selected checkpoints with a complete JSON report.

The selection law promotes the best eligible milestone, not necessarily the latest update. It checks success, physical visibility, target motion, learner motion, path length, turn caps, vertical-aim exercise, and finite transition evidence. The runner also records the exact world resident set, seed/hash identity, target program, stage contract, and evaluation placement.

`pursuit_run.py` is a study/entry-point layer. It consumes the reusable pursuit contract and skill-stage machinery; it should not be treated as the generic implementation of every Arena lesson.

