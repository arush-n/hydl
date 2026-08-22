"""What compute is actually available, and what a run is paying for.

Two things were invisible and both change how you read a result:

* **Which device the environment ran on.** A number produced on CPU and one
  produced on GPU are the same number, but the throughput behind them differs
  by an order of magnitude, and "training is slow" has a different cause on
  each.
* **Whether a run paid for a cold start.** Building a Region scene costs ~40 s
  and compiling the collector another ~60 s. Repeating a configuration should
  cost neither, and if it does, the scene cache is not being hit.

On Windows this reports **CPU** and that is not a misconfiguration to chase:
the JAX wheels for Windows are CPU-only. GPU here means WSL with CUDA JAX,
where the same work measured 1.75-1.96x against Windows CPU. Saying so beats
leaving someone to hunt for a driver that was never going to load.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _memory(device: Any) -> dict[str, Any]:
    """Per-device memory, when the backend exposes it. CPU does not."""

    stats = getattr(device, "memory_stats", None)
    if stats is None:
        return {}
    try:
        raw = stats() or {}
    except Exception:  # noqa: BLE001 - a backend without stats is normal
        return {}
    out: dict[str, Any] = {}
    for key, label in (("bytes_in_use", "in_use_mb"),
                       ("peak_bytes_in_use", "peak_mb"),
                       ("bytes_limit", "limit_mb")):
        if key in raw:
            out[label] = round(int(raw[key]) / (1024 ** 2), 1)
    return out


def status() -> dict[str, Any]:
    """Devices, backend, and how the scene cache is doing."""

    import jax

    from console.core.execution.runner import cache_state

    backend = jax.default_backend()
    devices = []
    for device in jax.devices():
        devices.append({
            "id": device.id,
            "kind": device.platform,
            "type": getattr(device, "device_kind", "unknown"),
            **_memory(device),
        })

    note = None
    if backend == "cpu":
        note = (
            "JAX is on CPU. On Windows that is the only option — the wheels "
            "are CPU-only, so there is no driver to fix. GPU means running "
            "under WSL with CUDA JAX, measured 1.75–1.96x faster there; "
            "Windows-native JAX measured 0.91x, i.e. slower than not using it."
        )

    return {
        "backend": backend,
        "devices": devices,
        "device_count": len(devices),
        "platform": platform.system(),
        "python": platform.python_version(),
        "jax": getattr(jax, "__version__", "unknown"),
        # A 64-bit default silently doubles memory and diverges from the Java
        # port, which matches JAX in float32 — measured ~800x further apart in
        # double where the expression cancels.
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "compilation_cache": (
            getattr(jax.config, "jax_compilation_cache_dir", None)
            or os.environ.get("HYTALERL_JAX_CACHE")
            or os.environ.get("JAX_COMPILATION_CACHE_DIR")
            or None
        ),
        "preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "default"),
        "scene_cache": cache_state(),
        "note": note,
    }
