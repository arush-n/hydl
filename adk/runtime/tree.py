"""JIT-safe selection for environment trees with shared provider leaves."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


ENVIRONMENT_STATE_SELECTOR_SCHEMA = (
    "hytalerl_adk_batch_owned_keep_shared_v1"
)


def select_environment_rows(mask: jax.Array, when_true: Any, when_false: Any):
    """Select batch-owned leaves and retain shared immutable leaves.

    Exact Region state contains per-environment mutable arrays alongside an
    immutable atlas and trace-time constants whose leading dimensions are a
    working-set capacity, not the environment batch. Per-row autoreset applies
    only to leaves with a leading environment axis. Shared leaves remain from
    the current state; providers must put mutable state on the batch axis.
    """

    selected = jnp.asarray(mask, dtype=jnp.bool_)
    if selected.ndim != 1:
        raise ValueError("environment row mask must have shape [batch]")

    def choose(true_value, false_value):
        true_array = jnp.asarray(true_value)
        false_array = jnp.asarray(false_value)
        if true_array.shape != false_array.shape:
            raise ValueError("environment PyTree leaf shape mismatch")
        if true_array.ndim == 0 or true_array.shape[0] != selected.shape[0]:
            return false_array
        expanded = selected.reshape(
            (selected.shape[0],) + (1,) * (true_array.ndim - 1)
        )
        return jnp.where(expanded, true_array, false_array)

    return jax.tree.map(choose, when_true, when_false)


def nonbatched_environment_leaves(
    state: Any,
    batch: int,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return paths/shapes incompatible with Gym PPO's current selector."""

    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    incompatible = []
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(state)
    for path, leaf in leaves_with_paths:
        shape = getattr(leaf, "shape", None)
        if shape is None:
            incompatible.append((jax.tree_util.keystr(path), ()))
            continue
        normalized_shape = tuple(int(size) for size in shape)
        if not normalized_shape or normalized_shape[0] != batch:
            incompatible.append(
                (jax.tree_util.keystr(path), normalized_shape)
            )
    return tuple(incompatible)


def require_gym_ppo_autoreset_compatible(state: Any, batch: int) -> None:
    """Fail before compilation if Gym PPO cannot select this state tree."""

    incompatible = nonbatched_environment_leaves(state, batch)
    if not incompatible:
        return
    preview = ", ".join(
        f"{path or '<root>'}:{shape}"
        for path, shape in incompatible[:8]
    )
    remainder = len(incompatible) - 8
    if remainder:
        preview += f", ... (+{remainder} more)"
    raise RuntimeError(
        "Gym PPO autoreset currently requires every environment-state leaf "
        f"to have leading batch axis {batch}; shared provider leaves: "
        f"{preview}. Raw JAX rollout/evaluation supports shared immutable "
        "leaves, but upstream PPO needs a state-selector port before this "
        "scene can train through the PPO facade."
    )


__all__ = [
    "ENVIRONMENT_STATE_SELECTOR_SCHEMA",
    "nonbatched_environment_leaves",
    "require_gym_ppo_autoreset_compatible",
    "select_environment_rows",
]
