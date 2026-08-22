"""Explicit host-side JAX runtime configuration for agent development."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any

import jax
import jaxlib


@dataclass(frozen=True, slots=True)
class JaxRuntimeSettings:
    """Applied persistent-cache settings and the selected default device."""

    enabled: bool
    compilation_cache_root: Path
    compilation_cache: Path | None
    runtime_namespace: str | None
    persistent_cache_min_compile_seconds: float
    compilation_cache_max_size_bytes: int
    device_kind: str
    disabled_reason: str | None


def configure_jax_compilation_cache(
    path: str | Path,
    *,
    min_compile_seconds: float = 1.0,
    max_size_bytes: int = -1,
    namespace_by_runtime: bool = True,
    allow_unsafe_windows_cpu_cache: bool = False,
) -> JaxRuntimeSettings:
    """Configure JAX's persistent executable cache at an explicit path.

    Call this before creating compiled rollouts or training steps. JAX owns
    cache keys and executable serialization; the ADK only validates the path
    and exposes the installed runtime's supported configuration fields.
    ``-1`` keeps JAX's unlimited-size default. Windows CPU persistence is
    disabled unless explicitly overridden because the installed XLA runtime
    has demonstrated a cross-process AOT target-feature mismatch; ordinary
    in-process ``jax.jit`` caching is unaffected.
    """

    if isinstance(path, Path):
        cache_path = path
    elif isinstance(path, str) and path.strip():
        cache_path = Path(path)
    else:
        raise TypeError("compilation cache path must be a path or string")
    if not isinstance(namespace_by_runtime, bool):
        raise TypeError("namespace_by_runtime must be bool")
    if not isinstance(allow_unsafe_windows_cpu_cache, bool):
        raise TypeError("allow_unsafe_windows_cpu_cache must be bool")

    if (
        isinstance(min_compile_seconds, bool)
        or not isinstance(min_compile_seconds, (int, float))
        or not math.isfinite(float(min_compile_seconds))
        or float(min_compile_seconds) < 0.0
    ):
        raise ValueError("min_compile_seconds must be finite and nonnegative")
    if (
        isinstance(max_size_bytes, bool)
        or not isinstance(max_size_bytes, int)
        or (max_size_bytes != -1 and max_size_bytes < 1)
    ):
        raise ValueError("max_size_bytes must be -1 or a positive integer")

    cache_root = cache_path.expanduser().resolve()
    if cache_root.exists() and not cache_root.is_dir():
        raise ValueError("compilation cache path is not a directory")
    device = jax.devices()[0]
    if (
        sys.platform == "win32"
        and device.platform == "cpu"
        and not allow_unsafe_windows_cpu_cache
    ):
        # JAX 0.9.2/XLA on this Windows CPU accepted cache keys produced by
        # the same runtime, then reported an AOT target mismatch for the
        # pseudo-features prefer-no-gather/prefer-no-scatter. XLA warns that
        # executing such an entry can SIGILL. Keep fast in-process JIT reuse,
        # but do not opt this runtime into unsafe cross-process executables.
        jax.config.update("jax_enable_compilation_cache", False)
        jax.config.update("jax_compilation_cache_dir", None)
        return JaxRuntimeSettings(
            enabled=False,
            compilation_cache_root=cache_root,
            compilation_cache=None,
            runtime_namespace=None,
            persistent_cache_min_compile_seconds=float(
                min_compile_seconds
            ),
            compilation_cache_max_size_bytes=max_size_bytes,
            device_kind=device.device_kind,
            disabled_reason=(
                "persistent executable cache disabled on Windows CPU after "
                "an XLA AOT target-feature mismatch; in-process jax.jit "
                "reuse remains enabled"
            ),
        )

    runtime_namespace = (
        _runtime_namespace(device) if namespace_by_runtime else None
    )
    cache_path = (
        cache_root / runtime_namespace
        if runtime_namespace is not None
        else cache_root
    )
    cache_path.mkdir(parents=True, exist_ok=True)

    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_compilation_cache_dir", str(cache_path))
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs",
        float(min_compile_seconds),
    )
    jax.config.update("jax_compilation_cache_max_size", max_size_bytes)
    return JaxRuntimeSettings(
        enabled=True,
        compilation_cache_root=cache_root,
        compilation_cache=cache_path,
        runtime_namespace=runtime_namespace,
        persistent_cache_min_compile_seconds=float(min_compile_seconds),
        compilation_cache_max_size_bytes=max_size_bytes,
        device_kind=device.device_kind,
        disabled_reason=None,
    )


def _runtime_namespace(device: Any) -> str:
    manifest = {
        "schema": "hytalerl_adk_jax_cache_runtime_v1",
        "python": list(sys.version_info[:3]),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "device_platform": device.platform,
        "device_kind": device.device_kind,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }
    payload = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{device.platform}-{digest}"


__all__ = [
    "JaxRuntimeSettings",
    "configure_jax_compilation_cache",
]
