"""Training and evaluation on exact WorldGen V2 tasks.

``worldgen_il``         end-to-end recurrent IL -> PPO on a WorldGen V2 task
``worldgen_benchmark``  compare that training across WorldGen V2 tasks
``worldgen_trace``      exact WorldGen traces mapped to PPO imitation batches
``minigames``           small custom games that verify generic IL composition
"""

from __future__ import annotations

from agents.ppo.worldgen import minigames as minigames
from agents.ppo.worldgen import worldgen_benchmark as worldgen_benchmark
from agents.ppo.worldgen import worldgen_il as worldgen_il
from agents.ppo.worldgen import worldgen_trace as worldgen_trace


__all__ = [
    "minigames",
    "worldgen_benchmark",
    "worldgen_il",
    "worldgen_trace",
]
