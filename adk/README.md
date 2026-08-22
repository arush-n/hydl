# Agent Development Kit

`adk` is the agent-facing SDK for building, collecting, evaluating, and
deploying policies against the HytaleRL environment. It preserves the
structured observation and action contracts while providing scene construction,
rollout, diagnostics, optional PPO tools, and the native-server boundary.

The kit depends on the nested Hytale gym and may adapt Arena through
[`arena.py`](arena.py). The dependency is deliberately one-way: ADK can import
Arena's task, training, evaluation, and imitation interfaces, while Arena never
imports ADK. The Console and concrete agents use both layers at their
respective boundaries.

## Entry points

- [`api.py`](api.py) exposes `AgentKit`, `AgentHandle`, scene construction,
  collection, training, evaluation, and checkpoint operations.
- [`policy/`](policy/README.md) defines the policy protocol and named
  observation/action surfaces.
- [`runtime/`](runtime/README.md) owns lossless stepping, episode boundaries,
  rollouts, and hierarchical collection.
- [`environments/`](environments/README.md) builds scenes and world providers;
  [`scenarios/`](scenarios/README.md) defines tasks and shaped objectives.
- [`probes/`](probes/README.md) and [`diagnostics/`](diagnostics/README.md)
  measure legality, liveness, failures, and trajectory structure.
- [`deploy/`](deploy/README.md) verifies and exports a trained policy for the
  native bridge.

For Arena task integration, start with [`arena.py`](arena.py). For the complete
training lesson used by the Console, start with [`arena/README.md`](../arena/README.md).
