"""Core layer of the console package.

Everything the console can *do* lives in one of five subpackages, each of whose
``__init__`` docstring names its modules:

``execution``
    Running things, and the compute they run on -- jobs, rollouts, self-play,
    checks, the GPU worker, device probing, and the live Hytale server.
``evidence``
    What a run leaves behind -- the artifact store, result normalization, the
    shared evidence contract, diagnostics, and the per-agent archive sync.
``catalog``
    Who can run and what they are -- discovered agents, profiles, identities.
``worlds``
    The world a run happens in -- worldgen previews, custom games and goals,
    terrain capture.
``telemetry``
    What the console can see while a run happens -- the live push stream,
    metric streams, and ADK/server surfaces.

``panels`` stays at this level on purpose: it is the registry an agent uses to
add a dashboard panel while the console is running, so it sits above the split
rather than inside it, and ``console/panels/*`` import it directly.

Import through the subpackage that owns the module::

    from console.core.execution import jobs
    from console.core.worlds import worldgen
"""

from __future__ import annotations
