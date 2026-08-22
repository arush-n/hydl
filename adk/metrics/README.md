# Trajectory metrics

`adk.metrics` computes pure statistics from recorded trajectories, including
engagement, damage efficiency, combos, punish rate, spacing, and whiffs. The
same functions can serve analysis or a reward term without maintaining a
second implementation.

It depends only on trajectory arrays and timing conventions. Start with
[`combat.py`](combat.py); no environment construction or optimizer is required.

For task-level reward laws, continue to [`arena/league/tasks/framework/`](../../arena/league/tasks/framework/README.md).
