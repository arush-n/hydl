"""Framework-neutral JAX policy primitives.

This module owns the reusable categorical and factored-action distribution
operations used by agents. It deliberately has no dependency on PPO or the
training package. Explicit factored transport is represented as int32 values
on the final head axis; legacy scalar packing remains a bounded compatibility
helper only.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array


class DenseParams(NamedTuple):
    """Weights and bias for one dense projection."""

    kernel: Array
    bias: Array


def sample_actions(
    key: jax.Array,
    logits: jax.Array,
    head_sizes: tuple[int, ...] = (),
    *,
    transport: str = "scalar",
) -> tuple[jax.Array, jax.Array]:
    """Sample one categorical or an explicit/legacy factored adapter."""

    if head_sizes:
        factors, log_probabilities = sample_action_factors(
            key,
            logits,
            head_sizes,
        )
        if transport == "factors":
            return factors, log_probabilities
        _validate_action_transport(transport)
        return pack_action_factors(factors, head_sizes), log_probabilities
    if transport != "scalar":
        raise ValueError("factor action transport requires factored head sizes")
    actions = jax.random.categorical(
        key,
        logits,
        axis=-1,
    ).astype(jnp.int32)
    log_probabilities = action_log_probabilities(logits, actions)
    return actions, log_probabilities


def sample_action_factors(
    key: jax.Array,
    logits: jax.Array,
    head_sizes: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    """Sample independent heads without coupling them into one integer."""

    _validate_head_sizes(head_sizes)
    keys = jax.random.split(key, len(head_sizes))
    heads = jnp.split(
        logits,
        tuple(_cumulative_sizes(head_sizes)[:-1]),
        axis=-1,
    )
    factors = jnp.stack(
        tuple(
            jax.random.categorical(
                head_key,
                head_logits,
                axis=-1,
            ).astype(jnp.int32)
            for head_key, head_logits in zip(keys, heads, strict=True)
        ),
        axis=-1,
    )
    return factors, action_factor_log_probabilities(
        logits,
        factors,
        head_sizes,
    )


def action_factor_log_probabilities(
    logits: jax.Array,
    factors: jax.Array,
    head_sizes: tuple[int, ...],
) -> jax.Array:
    """Sum log probabilities from an explicit last-axis head vector."""

    _validate_head_sizes(head_sizes)
    selected = jnp.asarray(factors, dtype=jnp.int32)
    expected_shape = logits.shape[:-1] + (len(head_sizes),)
    if selected.shape != expected_shape:
        raise ValueError(
            "action factors must match logits batch axes and head count: "
            f"{selected.shape} != {expected_shape}"
        )
    heads = jnp.split(
        logits,
        tuple(_cumulative_sizes(head_sizes)[:-1]),
        axis=-1,
    )
    return sum(
        action_log_probabilities(head, selected[..., index])
        for index, head in enumerate(heads)
    )


def pack_action_factors(
    factors: jax.Array,
    head_sizes: tuple[int, ...],
) -> jax.Array:
    """Pack a bounded head vector for the legacy scalar adapter."""

    _validate_head_sizes(head_sizes)
    _validate_scalar_action_capacity(head_sizes)
    selected = jnp.asarray(factors, dtype=jnp.int32)
    if selected.ndim < 1 or selected.shape[-1] != len(head_sizes):
        raise ValueError("action factors must end with one value per head")
    return _pack_action_factors(
        tuple(selected[..., index] for index in range(len(head_sizes))),
        head_sizes,
    )


def unpack_action_factors(
    actions: jax.Array,
    head_sizes: tuple[int, ...],
) -> jax.Array:
    """Unpack the legacy scalar adapter into an explicit head vector."""

    _validate_head_sizes(head_sizes)
    _validate_scalar_action_capacity(head_sizes)
    return jnp.stack(
        _unpack_action_factors(actions, head_sizes),
        axis=-1,
    )


def action_log_probabilities(
    logits: jax.Array,
    actions: jax.Array,
    head_sizes: tuple[int, ...] = (),
    *,
    transport: str = "scalar",
) -> jax.Array:
    if head_sizes:
        if transport == "factors":
            return action_factor_log_probabilities(
                logits,
                actions,
                head_sizes,
            )
        _validate_action_transport(transport)
        return action_factor_log_probabilities(
            logits,
            unpack_action_factors(actions, head_sizes),
            head_sizes,
        )
    if transport != "scalar":
        raise ValueError("factor action transport requires factored head sizes")
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(
        log_probabilities,
        actions[..., None],
        axis=-1,
    )[..., 0]


def categorical_entropy(
    logits: jax.Array,
    head_sizes: tuple[int, ...] = (),
) -> jax.Array:
    if head_sizes:
        heads = jnp.split(
            logits,
            tuple(_cumulative_sizes(head_sizes)[:-1]),
            axis=-1,
        )
        return sum(categorical_entropy(head) for head in heads)
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    probabilities = jnp.exp(log_probabilities)
    return -jnp.sum(probabilities * log_probabilities, axis=-1)


def _cumulative_sizes(head_sizes: tuple[int, ...]) -> tuple[int, ...]:
    total = 0
    cumulative = []
    for size in head_sizes:
        total += size
        cumulative.append(total)
    return tuple(cumulative)


def _validate_head_sizes(head_sizes: tuple[int, ...]) -> None:
    if not head_sizes:
        raise ValueError("head_sizes must not be empty")
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size < 1
        for size in head_sizes
    ):
        raise ValueError("head sizes must be positive integers")


def _validate_scalar_action_capacity(
    head_sizes: tuple[int, ...],
) -> None:
    combination_count = 1
    maximum = 2**31 - 1
    for size in head_sizes:
        if combination_count > maximum // size:
            raise OverflowError(
                "factored actions exceed int32 scalar transport capacity; "
                "use explicit per-head transport"
            )
        combination_count *= size


def _validate_action_transport(transport: str) -> None:
    if transport not in {"scalar", "factors"}:
        raise ValueError("action transport must be 'scalar' or 'factors'")


def _pack_action_factors(
    factors: tuple[jax.Array, ...],
    head_sizes: tuple[int, ...],
) -> jax.Array:
    packed = jnp.zeros_like(factors[0], dtype=jnp.int32)
    for factor, size in zip(factors, head_sizes, strict=True):
        packed = packed * jnp.int32(size) + factor
    return packed


def _unpack_action_factors(
    actions: jax.Array,
    head_sizes: tuple[int, ...],
) -> tuple[jax.Array, ...]:
    remainder = jnp.asarray(actions, dtype=jnp.int32)
    reversed_factors = []
    for size in reversed(head_sizes):
        reversed_factors.append(remainder % jnp.int32(size))
        remainder = remainder // jnp.int32(size)
    return tuple(reversed(reversed_factors))


__all__ = [
    "DenseParams",
    "action_factor_log_probabilities",
    "action_log_probabilities",
    "categorical_entropy",
    "pack_action_factors",
    "sample_action_factors",
    "sample_actions",
    "unpack_action_factors",
]

