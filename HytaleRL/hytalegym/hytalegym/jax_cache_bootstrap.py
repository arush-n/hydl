"""Configure persistent JAX compilation caching before JAX-heavy imports.

Import :data:`PERSISTENT_COMPILATION_CACHE` before importing
``hytalegym.jax`` or any module that performs eager JAX work.  JAX binds its
file-cache object once; changing only the public config value after that point
can make a report name a directory that the process never uses.
"""

from __future__ import annotations

import hashlib
import os
import platform
from pathlib import Path
import sys

import jax
import jaxlib


def _materialized_cache() -> tuple[Path | None, bool]:
    """Return JAX's actual one-shot cache path and initialization state."""

    try:
        from jax._src import compilation_cache as internal_cache
    except ImportError:
        return None, False

    cache = getattr(internal_cache, "_cache", None)
    initialized = bool(getattr(internal_cache, "_cache_initialized", cache is not None))
    if cache is None:
        return None, initialized
    path = getattr(cache, "_path", None)
    if path is None:
        return None, initialized
    return Path(path).expanduser().resolve(), True


def configure_persistent_compilation_cache(
    default_root: str | Path | None = None,
) -> Path | None:
    """Enable executable reuse without crossing CPU, GPU, or JAX identities.

    The normal path must run before any eager JAX compilation.  If another
    import already materialized JAX's one-shot cache, the function leaves it
    untouched and returns its real directory so provenance cannot point at an
    unused replacement.
    """

    override = os.environ.get("HYTALERL_JAX_CACHE")
    backend = jax.default_backend()
    devices = jax.devices()
    active_directory, cache_initialized = _materialized_cache()
    if active_directory is not None:
        active_directory.mkdir(parents=True, exist_ok=True)
        return active_directory
    if cache_initialized:
        # JAX has irreversibly initialized an unsupported or unavailable cache.
        # Reporting a newly configured directory here would be false.
        return None
    if backend == "cpu" and override is None:
        jax.config.update("jax_compilation_cache_dir", None)
        return None

    standard = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    cache_root = Path(
        override
        or standard
        or default_root
        or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "hytalerl"
        / "jax"
    ).expanduser()
    target = "|".join(
        (
            platform.platform(),
            platform.processor(),
            ",".join(
                sorted(f"{device.platform}:{device.device_kind}" for device in devices)
            ),
            os.environ.get("XLA_FLAGS", ""),
        )
    )
    identity = "-".join(
        (
            sys.platform,
            platform.machine().lower(),
            backend,
            f"jax-{jax.__version__}",
            f"jaxlib-{jaxlib.__version__}",
            f"target-{hashlib.sha256(target.encode()).hexdigest()[:16]}",
        )
    )
    directory = (cache_root / identity).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_compilation_cache_dir", str(directory))
    return directory


# Campaign modules import this symbol before any JAX-heavy dependency.  Keeping
# the side effect here avoids an assignment between import statements and makes
# the ordering requirement reusable by command-line and programmatic callers.
PERSISTENT_COMPILATION_CACHE = configure_persistent_compilation_cache()


__all__ = [
    "PERSISTENT_COMPILATION_CACHE",
    "configure_persistent_compilation_cache",
]
