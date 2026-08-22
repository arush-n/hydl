# Validation and identity

`adk.validation` checks that a scene, native capture, policy bundle, and
episode lifecycle agree on the same contract. It provides fail-closed identity,
reset, lifecycle, and region validation rather than a task score.

It depends on the HytaleGym contract and native evidence boundaries and is used
by scene construction, evaluation, deployment, and Console evidence. Start with
[`identity.py`](identity.py), [`lifecycle.py`](lifecycle.py), and
[`verdict.py`](verdict.py).

For task-level JAX validation, continue to
[`arena/league/tasks/framework/`](../../arena/league/tasks/framework/README.md).
