# Lossless development objects

`adk.development` names the lossless frames, structured actions, transitions,
and diagnostics that agent authors handle at the JAX/native boundary. It keeps
the upstream arrays and metadata intact rather than projecting them into a
task-specific feature vector.

The module depends on HytaleGym observation and action contracts and is consumed
by the ADK runtime and architecture layers. It does not perform collection or
select a training objective.

The implementation is in [`__init__.py`](__init__.py); the public runtime
consumer is [`runtime/`](../runtime/README.md).
