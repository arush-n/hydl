# Built-in agent families

This package contains ADK-owned agent specifications and family registration.
It describes which policy modules can be loaded; it does not own Arena tasks or
the native server loop.

The public family currently lives under [`combat/`](combat). It is resolved by
[`adk/api.py`](../api.py) when an `AgentSpec` names a family. Keep family
configuration separate from the generic [`policy/`](../policy/README.md) and
[`runtime/`](../runtime/README.md) contracts.

For concrete project agents and Console profiles, continue to
[`agents/`](../../agents/README.md).
