import os
import platform
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for path in (_ROOT, _ROOT / "HytaleRL" / "hytalegym"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _shared_jax_cache_directory() -> Path:
    """Resolve one backend-compatible reusable JAX compilation cache.

    Every pytest process otherwise recompiles from scratch; the suite is
    compile-bound at roughly ten seconds per test.

    The partitioning key is kept byte-identical to the Gym suite's
    (``hytalegym/tests/conftest.py``) so both suites share entries instead of
    each warming its own copy.  XLA cache entries are not portable across
    native Windows CPU and WSL CUDA, so an unpartitioned root can offer one
    lane an executable the other cannot run.  ``HYTALERL_JAX_CACHE`` stays an
    explicit clean-room override for measurement runs.
    """

    import jax
    import jaxlib

    override = os.environ.get("HYTALERL_JAX_CACHE")
    root = (
        Path(override).expanduser()
        if override
        else _ROOT / "HytaleRL" / ".codex-local" / "jax-cache"
    )
    if not root.is_absolute():
        root = _ROOT / root
    compatibility_key = "-".join(
        (
            sys.platform,
            platform.machine().lower(),
            jax.default_backend(),
            f"jax-{jax.__version__}",
            f"jaxlib-{jaxlib.__version__}",
        )
    )
    return (root / compatibility_key).resolve()


# Configure before any test module imports jax and traces a computation.
# ``jax.config.update`` is used rather than an environment variable because the
# variable is only read at jax import time, and conftest ordering does not
# guarantee we run first.
import jax  # noqa: E402  (path setup above must precede this import)

JAX_COMPILATION_CACHE = _shared_jax_cache_directory()
JAX_COMPILATION_CACHE.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(JAX_COMPILATION_CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
