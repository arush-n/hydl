"""HytaleRL — Gymnasium-compatible RL environments for Hytale."""

from hytalegym.envs.hytale_env import HytaleEnv
from hytalegym.envs.registration import register_envs

__version__ = "0.1.0"
__all__ = ["HytaleEnv", "register_envs"]

# Auto-register environments on import
register_envs()
