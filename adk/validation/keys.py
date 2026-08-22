"""Batch-independent key streams.

The obvious way to build per-tick, per-environment keys is:

    jax.random.split(key, TICKS * BATCH).reshape(TICKS, BATCH)   # WRONG

Environment ``b``'s stream then depends on **BATCH**, so adding a profile to a
comparison silently rerolls every other environment. Two runs at the same seed
with different batch composition are not comparable, and the difference looks
like a behavioural finding.

That is not hypothetical. "Abilities fire on release, never while held" was
measured at 3 profiles and retracted at 9: the held arm went from *1 start,
0 completions* to *61 starts, 60 completions* inside a single episode. The
mechanism had not changed; the keys had.

``stream()`` derives environment ``b``'s keys from ``b`` alone, so a run with
9 profiles reproduces the first 3 environments of a run with 3.
"""

from __future__ import annotations

import jax


def _as_key(seed: int | jax.Array) -> jax.Array:
    """Accept either a PRNG key or a plain seed.

    Detected by **dtype**, not ``isinstance``: under ``jax.jit`` a traced
    integer is also a ``jax.Array``, so an isinstance check silently passes a
    seed where a key is expected and fails with "expected key_data.ndim >= 1".
    """

    dtype = getattr(seed, "dtype", None)
    if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
        return seed
    return jax.random.key(seed)


def env_key(seed: int | jax.Array, index: int) -> jax.Array:
    """The root key for environment ``index``, independent of batch size."""

    return jax.random.fold_in(_as_key(seed), index)


def reset_keys(seed: int | jax.Array, batch: int) -> jax.Array:
    """One reset key per environment, stable as ``batch`` grows.

    ``reset_keys(11, 9)[:3]`` equals ``reset_keys(11, 3)``.
    """

    return jax.numpy.stack([env_key(seed, i) for i in range(batch)])


def stream(seed: int | jax.Array, ticks: int, batch: int) -> jax.Array:
    """``(ticks, batch)`` keys where column ``b`` depends only on ``b``.

    Use this for the ``xs`` of a rollout ``lax.scan``. Adding an environment
    leaves every existing column byte-identical, so a comparison across batch
    compositions measures the change under test rather than the RNG.
    """

    columns = [jax.random.split(env_key(seed, i), ticks) for i in range(batch)]
    return jax.numpy.stack(columns, axis=1)
