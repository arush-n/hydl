"""Process defaults for HytaleGym JAX workloads on WSL/CUDA."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import MutableMapping


def configure_runtime_environment(
    environ: MutableMapping[str, str] | None = None,
    *,
    release: str | None = None,
    home: str | Path | None = None,
    gpu_available: bool | None = None,
    profile: str | None = None,
    vram_fraction: float = 0.90,
) -> dict[str, str]:
    """Apply safe WSL/GPU defaults without overriding explicit settings."""

    env = os.environ if environ is None else environ
    kernel = platform.release() if release is None else release
    if not (env.get("WSL_DISTRO_NAME") or "microsoft" in kernel.lower()):
        return {}

    root = Path.home() if home is None else Path(home)
    selected_profile = profile or env.get("HYTALERL_JAX_RUNTIME_PROFILE", "shared")
    if selected_profile not in {"shared", "throughput"}:
        raise ValueError("JAX runtime profile must be shared or throughput")
    if not 0.0 < vram_fraction <= 1.0:
        raise ValueError("vram_fraction must be in (0, 1]")
    defaults = {
        "HYTALERL_JAX_RUNTIME_PROFILE": selected_profile,
        "XLA_PYTHON_CLIENT_PREALLOCATE": (
            "true" if selected_profile == "throughput" else "false"
        ),
        "HYTALERL_JAX_CACHE": str(root / ".cache" / "hytalerl" / "jax"),
    }
    if selected_profile == "throughput":
        defaults["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(vram_fraction)
    has_gpu = Path("/dev/dxg").exists() if gpu_available is None else gpu_available
    if has_gpu:
        defaults["JAX_PLATFORMS"] = "cuda"
    for name, value in defaults.items():
        env.setdefault(name, value)
    return {name: env[name] for name in defaults}


def configure_training_runtime(
    environ: MutableMapping[str, str] | None = None,
    *,
    vram_fraction: float = 0.90,
) -> dict[str, str]:
    """Select the dedicated high-VRAM profile before importing JAX."""

    return configure_runtime_environment(
        environ,
        profile="throughput",
        vram_fraction=vram_fraction,
    )


__all__ = ["configure_runtime_environment", "configure_training_runtime"]
