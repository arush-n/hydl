"""Runtime defaults that keep JAX compilation off slow WSL mounts."""

from __future__ import annotations

import os
from pathlib import Path
import platform


def default_compilation_cache() -> Path:
    """Use native WSL storage unless the operator selects another cache."""

    override = os.environ.get("HYTALERL_PPO_JAX_CACHE") or os.environ.get(
        "HYTALERL_JAX_CACHE"
    )
    if override:
        return Path(override).expanduser()
    if platform.system() == "Linux" and "microsoft" in platform.release().lower():
        return Path.home() / ".cache/hytalerl/agents-ppo-worldgen-il"
    return Path(".jax_cache/agents-ppo-worldgen-il")


__all__ = ["default_compilation_cache"]
