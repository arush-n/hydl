"""Public recurrent IL -> PPO agent for exact WorldGen V2 tasks."""

from __future__ import annotations

from agents.ppo.worldgen.worldgen_il import WorldgenILPPOAgent, WorldgenILPPOSettings


class PPOAgent(WorldgenILPPOAgent):
    """Train any compatible Arena task from exact JAX expert transitions."""


PPOSettings = WorldgenILPPOSettings

# Keep imports used by older agents working without retaining the open-flat
# implementation. Both names now resolve to the production WorldGen path.
SimplePPOAgent = PPOAgent
SimplePPOSettings = PPOSettings

__all__ = [
    "PPOAgent",
    "PPOSettings",
    "SimplePPOAgent",
    "SimplePPOSettings",
]
