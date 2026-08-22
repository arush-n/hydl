# ADK core

`adk.core` provides the declarative identity and lifecycle primitives beneath
the public `AgentKit` API. Specs name agents and scenes; the registry resolves
families; lifecycle helpers bind parameters and checkpoint state.

This layer depends on the ADK contracts but not on Arena, Console, or a
particular optimizer. Higher-level callers normally enter through
[`adk/api.py`](../api.py) rather than importing the core modules directly.

The main pieces are [`spec.py`](spec.py), [`registry.py`](registry.py), and
[`lifecycle/`](lifecycle).
