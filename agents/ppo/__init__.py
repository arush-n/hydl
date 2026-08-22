"""Recurrent IL -> PPO agents built on the public HytaleRL ADK."""

from adk import _runtime_environment as _runtime_environment

from agents.ppo.agent import (
    PPOAgent,
    PPOSettings,
    SimplePPOAgent,
    SimplePPOSettings,
)

__all__ = [
    "PPOAgent",
    "PPOSettings",
    "SimplePPOAgent",
    "SimplePPOSettings",
    "WorldgenBenchmarkSettings",
    "WorldgenILPPOAgent",
    "WorldgenILPPOSettings",
    "run_worldgen_benchmark",
]


def __getattr__(name: str):
    if name in {"WorldgenBenchmarkSettings", "run_worldgen_benchmark"}:
        from agents.ppo.worldgen import worldgen_benchmark

        return (
            worldgen_benchmark.run_benchmark
            if name == "run_worldgen_benchmark"
            else worldgen_benchmark.WorldgenBenchmarkSettings
        )
    if name in {"WorldgenILPPOAgent", "WorldgenILPPOSettings"}:
        from agents.ppo.worldgen import worldgen_il

        return getattr(worldgen_il, name)
    raise AttributeError(name)
