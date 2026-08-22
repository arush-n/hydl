"""Executable training entrypoints.

Each module here has a ``main()`` and is invoked with ``python -m``. They own
argument parsing, artifact layout and evaluation scheduling; the reward law they
train against lives in :mod:`arena.training.contracts`.

``pursuit_run``
    ``python -m arena.training.runs.pursuit_run`` -- trains and evaluates
    recurrent pursuit against a sampled actor in a JAX Region world.

Nothing is re-exported here on purpose. Importing an entrypoint from its own
package makes ``python -m`` execute the module twice -- once as
``arena.training.runs.<name>`` and again as ``__main__`` -- which duplicates
module-level state and emits a ``RuntimeWarning``. Import the module directly:

    from arena.training.runs.pursuit_run import pursuit_launch_options
"""

from __future__ import annotations

__all__: list[str] = []
