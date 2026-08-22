"""Compatibility import for the early JAX cache bootstrap."""

from __future__ import annotations

from hytalegym.jax_cache_bootstrap import (
    PERSISTENT_COMPILATION_CACHE,
    configure_persistent_compilation_cache,
)


__all__ = [
    "PERSISTENT_COMPILATION_CACHE",
    "configure_persistent_compilation_cache",
]
