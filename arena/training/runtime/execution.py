"""Device-wide execution primitives shared by every JAX trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp


REPLICATED_TRAINING_SCHEMA = "arena_replicated_training_v1"


@dataclass(frozen=True, slots=True)
class ReplicatedExecution:
    """One compiled program over compatible independent training replicas."""

    step: Callable[[Any, jax.Array], tuple[Any, Any]]
    replicas: int
    contract_sha256: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.replicas, bool) or self.replicas < 1:
            raise ValueError("replicas must be a positive integer")

    def __call__(self, state: Any, keys: jax.Array):
        if not keys.shape or keys.shape[0] != self.replicas:
            raise ValueError("keys must have one leading row per replica")
        return self.step(state, keys)

    def describe(self) -> dict[str, Any]:
        return {
            "schema": REPLICATED_TRAINING_SCHEMA,
            "mode": "jax_vmap",
            "replicas": self.replicas,
            "contract_sha256": self.contract_sha256,
            "single_device": True,
        }


def stack_replicas(values: Sequence[Any]) -> Any:
    """Add a leading replica axis to identical JAX pytrees."""

    rows = tuple(values)
    if not rows:
        raise ValueError("at least one replica is required")
    structure = jax.tree.structure(rows[0])
    if any(jax.tree.structure(row) != structure for row in rows[1:]):
        raise ValueError("replica pytrees differ")

    def stack(*leaves):
        arrays = tuple(jnp.asarray(leaf) for leaf in leaves)
        reference = arrays[0]
        if any(
            array.shape != reference.shape or array.dtype != reference.dtype
            for array in arrays[1:]
        ):
            raise ValueError("replica leaf shapes or dtypes differ")
        return jnp.stack(arrays)

    return jax.tree.map(stack, *rows)


def split_replicas(value: Any, replicas: int) -> tuple[Any, ...]:
    """Remove a leading replica axis without copying values to the host."""

    if isinstance(replicas, bool) or replicas < 1:
        raise ValueError("replicas must be a positive integer")
    leaves = jax.tree.leaves(value)
    if any(getattr(leaf, "shape", ())[:1] != (replicas,) for leaf in leaves):
        raise ValueError("replicated pytree has an incorrect leading axis")
    return tuple(
        jax.tree.map(lambda leaf, index=index: leaf[index], value)
        for index in range(replicas)
    )


def repeated_keys(key: jax.Array, replicas: int) -> jax.Array:
    """Broadcast one PRNG key for common-random-number comparisons."""

    key = jnp.asarray(key)
    if key.shape not in {(), (2,)}:
        raise ValueError("key must be one JAX PRNG key")
    return jnp.broadcast_to(key, (replicas,) + key.shape)


def make_replicated_step(
    step: Callable[[Any, jax.Array], tuple[Any, Any]],
    replicas: int,
    *,
    contract_sha256: str | None = None,
    compile: bool = True,
) -> ReplicatedExecution:
    """Apply ``vmap`` to an existing pure JAX train step."""

    if not callable(step):
        raise TypeError("step must be callable")
    if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 1:
        raise ValueError("replicas must be a positive integer")
    batched = jax.vmap(step, in_axes=(0, 0))
    return ReplicatedExecution(
        jax.jit(batched) if compile else batched,
        replicas,
        contract_sha256,
    )


def recommended_replicas(
    *,
    bytes_limit: int,
    measured_peak_bytes: int | None,
    maximum: int,
    vram_fraction: float = 0.90,
) -> int:
    """Size the next full run from a previous full run's VRAM peak."""

    if bytes_limit < 1 or maximum < 1:
        raise ValueError("bytes_limit and maximum must be positive")
    if not 0.0 < vram_fraction <= 1.0:
        raise ValueError("vram_fraction must be in (0, 1]")
    if measured_peak_bytes is None:
        return min(2, maximum)
    if measured_peak_bytes < 1:
        raise ValueError("measured_peak_bytes must be positive or None")
    return max(
        1,
        min(maximum, int(bytes_limit * vram_fraction) // measured_peak_bytes),
    )


__all__ = [
    "REPLICATED_TRAINING_SCHEMA",
    "ReplicatedExecution",
    "make_replicated_step",
    "recommended_replicas",
    "repeated_keys",
    "split_replicas",
    "stack_replicas",
]
