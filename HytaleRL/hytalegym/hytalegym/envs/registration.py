"""Register HytaleRL environments with Gymnasium."""

import gymnasium as gym

_REGISTERED = False

TASKS = {
    "HytaleRL-v0": {"task": "survive", "max_episode_steps": 6000},
    "HytaleSurvive-v0": {"task": "survive", "max_episode_steps": 6000},
    "HytaleMineAdamantite-v0": {"task": "mine_adamantite", "max_episode_steps": 12000},
    "HytaleKillTrork-v0": {"task": "kill_trork", "max_episode_steps": 4000},
    "HytaleBuildHouse-v0": {"task": "build_house", "max_episode_steps": 8000},
    "HytaleNavigate-v0": {"task": "navigate", "max_episode_steps": 4000},
    "HytaleBaseBuilder-v0": {"task": "base_builder", "max_episode_steps": 24000},
}


def register_envs():
    """Register all HytaleRL environments with Gymnasium."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    for env_id, kwargs in TASKS.items():
        gym.register(
            id=env_id,
            entry_point="hytalegym.envs.hytale_env:HytaleEnv",
            kwargs=kwargs,
            max_episode_steps=kwargs["max_episode_steps"],
        )
