"""The agent's own recent actions, as the ``act`` tokens of ``skill_tokens_v3`` read them (``docs/PERCEPTION.md``
section 10, decision 3; owner, 2026-10-20 night).

The engine never shows how long a button has been held, yet a charge, a bow's draw and a battleaxe's release all turn
on it. The agent knows what it pressed, so the neural layer carries it: per action head of the gen-7 layout, the
choice of the last tick, whether it changed on that tick, and for how many ticks it has stood (neutral included). The
carry is a NamedTuple of arrays, updated once per tick from the action the agent sent, inside ``jit``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract.surface import head_layout

HEADS = tuple(head_layout("gen7"))
HEAD_SIZES = np.asarray([head.size for head in HEADS], np.int32)
HEAD_NEUTRAL = np.asarray([head.neutral for head in HEADS], np.int32)
HELD_HORIZON_TICKS: int = 90
"""``held_ticks`` saturates here (3 s at 30 Hz): log1p(ticks) / log1p(90), clipped to 1."""


class ActionHistory(NamedTuple):
    """Per agent and head (gen-7 order): the last choice int32[B, 21], whether it changed on its tick bool[B, 21], and
    how many ticks it has stood int32[B, 21]."""

    last_choice: jax.Array
    changed: jax.Array
    held_ticks: jax.Array


def initial_history(batch: int) -> ActionHistory:
    """Every head neutral, unchanged, held for 0 ticks."""

    neutral = jnp.broadcast_to(jnp.asarray(HEAD_NEUTRAL), (batch, len(HEADS)))
    return ActionHistory(neutral, jnp.zeros((batch, len(HEADS)), jnp.bool_), jnp.zeros((batch, len(HEADS)), jnp.int32))


def update_history(history: ActionHistory, factors: jax.Array, reset: jax.Array | None = None) -> ActionHistory:
    """The history after the agent sent ``factors`` int32[B, 21] (the gen-7 heads); lanes where ``reset`` bool[B] is
    True start over from :func:`initial_history` first (a new episode)."""

    if reset is not None:
        fresh = initial_history(factors.shape[0])
        history = jax.tree_util.tree_map(lambda old, new: jnp.where(reset[:, None], new, old), history, fresh)
    factors = jnp.asarray(factors, jnp.int32)
    changed = factors != history.last_choice
    held = jnp.where(changed, 1, history.held_ticks + 1)
    return ActionHistory(factors, changed, jnp.minimum(held, 1 << 20))


def act_features(history: ActionHistory) -> jax.Array:
    """The ``act`` token rows float32[B, 21, 26] (``contract/tokens_v3.ActTokenV3``): the head's one-hot, the choice
    as a fraction of its choices, neutral, changed, held ticks, marked 0."""

    batch = history.last_choice.shape[0]
    heads = len(HEADS)
    one_hot = jnp.broadcast_to(jnp.eye(heads, dtype=jnp.float32), (batch, heads, heads))
    fraction = history.last_choice / jnp.maximum(jnp.asarray(HEAD_SIZES) - 1, 1)
    neutral = history.last_choice == jnp.asarray(HEAD_NEUTRAL)
    held = jnp.minimum(jnp.log1p(history.held_ticks.astype(jnp.float32)) / np.log1p(HELD_HORIZON_TICKS), 1.0)
    columns = jnp.stack((fraction.astype(jnp.float32), neutral.astype(jnp.float32),
                         history.changed.astype(jnp.float32), held, jnp.zeros_like(held)), axis=-1)
    return jnp.concatenate((one_hot, columns), axis=-1)


__all__ = ["HEADS", "HEAD_NEUTRAL", "HEAD_SIZES", "HELD_HORIZON_TICKS", "ActionHistory", "act_features",
           "initial_history", "update_history"]
