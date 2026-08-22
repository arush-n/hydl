# Diagnostics

`adk.diagnostics` decodes failure flags, records traced trajectories, and
renders compact views of observations and scene state. It is safe to use during
JAX collection when the selected recorder is trace-compatible, while rendering
and explanation helpers are host-side operations.

It depends on the HytaleGym diagnostic fields and does not depend on an
optimizer or task registry. Start with [`failures.py`](failures.py),
[`trace.py`](trace.py), and [`render.py`](render.py).

For active scene checks, continue to [`probes/`](../probes/README.md).
