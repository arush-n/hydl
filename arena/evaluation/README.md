# Evaluation

`arena.evaluation` contains the promotion gate for policies that have already been trained. It is intentionally algorithm-neutral: the caller supplies exact episode counts for each policy arm.

## `replication.py`

The main records are:

- `ArmResult`: successes and episodes for one policy arm.
- `EvaluationReplicate`: one independently trained seed evaluated on one named task and held-out split.
- `PromotionGate`: thresholds for task count, training-seed count, episodes, success, and advantage.
- `assess_promotion()`: produces `insufficient`, `fail`, or `pass` plus task-wise diagnostics.

The default gate requires at least three independently trained seeds per task, at least 64 episodes per arm, a 95% lower confidence bound of 70% success, and a five-point advantage over the configured baselines. Each named task must pass independently; vectorized environment lanes are episodes, not independent training replicas.

An incumbent can be added as a non-regression arm. WorldGen seed strata should be reported for diagnosis, but they do not replace independent training seeds as the replication unit.

## What this package does not do

It does not launch a collector, train a policy, or infer terminal outcomes from aggregate `done` flags. The caller must provide exact success/death counts and the correct held-out split. This keeps promotion evidence auditable and prevents pooled easy tasks from hiding a regression on a hard task.

For the training machinery that produces candidates for this gate, continue to
[`training/`](../training/README.md). For the task and goal vocabulary that the
gate evaluates, continue to [`league/tasks/`](../league/tasks/README.md).
