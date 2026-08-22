"""Running things, and the compute they run on.

``jobs``       long-running training jobs, launched and survivable
``runner``     parameterised rollouts for the debug console
``selfplay``   self-play as a runnable job
``checks``     named checks the console can launch and remember
``inflight``   read staged run scripts without executing them
``gpu_worker`` one reusable WSL process for serial CUDA jobs
``devices``    what compute is available and what a run is paying for
``runners``    which compute this machine can actually train on, probed
``hytale``     launching the real Hytale server and talking to its console
"""

from __future__ import annotations

from console.core.execution import checks as checks
from console.core.execution import devices as devices
from console.core.execution import gpu_worker as gpu_worker
from console.core.execution import hytale as hytale
from console.core.execution import inflight as inflight
from console.core.execution import jobs as jobs
from console.core.execution import runner as runner
from console.core.execution import runners as runners
from console.core.execution import selfplay as selfplay


__all__ = [
    "checks",
    "devices",
    "gpu_worker",
    "hytale",
    "inflight",
    "jobs",
    "runner",
    "runners",
    "selfplay",
]
