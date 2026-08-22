"""Arena: training games, curriculum ladders and league bookkeeping.

The arena ships a *framework*, not a task list. A game is whatever you
register; the curriculum and league machinery cannot tell the difference
between a game that shipped here and one written this afternoon.

Start from ``arena/tasks/TEMPLATE.py``.
"""

from arena.runtime import configure_runtime_environment

configure_runtime_environment()

__all__ = [
    "component",
    "curriculum",
    "evaluation",
    "jax_contract",
    "jax_env",
    "imitation",
    "league",
    "manager",
    "params",
    "publication_worlds",
    "runtime",
    "tasks",
    "training",
    "worlds",
]
