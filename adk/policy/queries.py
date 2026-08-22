"""Ask the observation questions: nearest, strongest, how many, is there one.

Every agent needs "where is the closest hostile" or "is there a hazard in front
of me", and every agent currently rewrites it. The rewrite is easy to get
wrong in one specific way:

    jnp.argmin(entities.values[..., distance])     # WRONG

A padded slot holds arbitrary data -- often zero -- so a naive ``argmin`` over
distance reliably returns an entity that does not exist, sitting at range zero.
The bug is silent: training proceeds, the policy learns to react to a phantom.
Every reduction here substitutes the identity element at masked slots first, so
padding can never win.

All of it is vectorized over the batch, jit-safe, and free of host
synchronization. Costs are linear in the set's *capacity*, which is a compile
-time constant (16 entities, 48 terrain patches, 44 world tokens), so these are
fixed-cost operations rather than data-dependent loops.

Results are returned as arrays including an explicit ``found`` flag, never as
Python values -- a query on an empty set must stay traceable, so it reports
"nothing matched" rather than raising.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from adk.policy.fields import field, field_index
from adk.policy.observations import TokenSet


class Selection(NamedTuple):
    """Slots picked out of a set, with whether anything was actually there.

    ``index`` is meaningless where ``found`` is false; it is clamped to a valid
    position so downstream gathers stay in bounds rather than producing
    undefined behaviour.
    """

    index: jax.Array
    value: jax.Array
    found: jax.Array

    def gather(self, tokens: TokenSet) -> jax.Array:
        """Pull the full feature rows for the selected slots.

        Handles both shapes this module produces: a single pick from
        :func:`nearest`/:func:`extreme` (rank ``batch``) and a ranked set from
        :func:`top_k` (rank ``batch, k``).  The result always carries an
        explicit selection axis, so downstream code does not branch on which
        query produced it.
        """

        indices = self.index
        if indices.ndim == tokens.values.ndim - 2:
            indices = indices[..., None]
        return jnp.take_along_axis(tokens.values, indices[..., None], axis=-2)


def _masked_for_extreme(
    values: jax.Array,
    mask: jax.Array,
    *,
    largest: bool,
) -> jax.Array:
    """Replace padded slots with the identity so they never win."""

    fill = -jnp.inf if largest else jnp.inf
    return jnp.where(mask, values, fill)


def extreme(
    tokens: TokenSet,
    group: str,
    name: str,
    *,
    largest: bool = False,
) -> Selection:
    """The slot with the smallest (or largest) value of a named field.

    ``largest=False`` is the common case: nearest by ``distance``.
    """

    values = field(tokens, group, name)
    candidates = _masked_for_extreme(values, tokens.mask, largest=largest)
    index = (
        jnp.argmax(candidates, axis=-1)
        if largest
        else jnp.argmin(candidates, axis=-1)
    )
    found = jnp.any(tokens.mask, axis=-1)
    chosen = jnp.take_along_axis(values, index[..., None], axis=-1)[..., 0]
    return Selection(
        index=jnp.where(found, index, 0),
        value=jnp.where(found, chosen, jnp.nan),
        found=found,
    )


def nearest(tokens: TokenSet, group: str, *, name: str = "distance") -> Selection:
    """The closest occupied slot.

    Most set groups publish a ``distance`` column; pass ``name`` for those that
    measure closeness differently.
    """

    return extreme(tokens, group, name, largest=False)


def top_k(
    tokens: TokenSet,
    group: str,
    name: str,
    k: int,
    *,
    largest: bool = True,
) -> Selection:
    """The ``k`` highest (or lowest) slots by a named field.

    ``found`` is per-rank, so a set holding fewer than ``k`` occupied slots
    reports exactly which ranks are real.
    """

    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k < 1:
        raise ValueError("k must be positive")
    capacity = tokens.values.shape[-2]
    if k > capacity:
        raise ValueError(f"k={k} exceeds the set capacity of {capacity}")

    values = field(tokens, group, name)
    candidates = _masked_for_extreme(values, tokens.mask, largest=largest)
    ordered = candidates if largest else -candidates
    picked, index = jax.lax.top_k(ordered, k)
    occupied = jnp.sum(tokens.mask.astype(jnp.int32), axis=-1, keepdims=True)
    ranks = jnp.arange(k, dtype=jnp.int32)
    return Selection(
        index=index,
        value=picked if largest else -picked,
        found=ranks < occupied,
    )


def matching(
    tokens: TokenSet,
    group: str,
    name: str,
    *,
    at_least: float | None = None,
    at_most: float | None = None,
) -> jax.Array:
    """A mask of occupied slots whose named field is within bounds.

    Compose it with :func:`count` or pass it as a new ``TokenSet`` mask to
    chain queries -- "hostile entities closer than 5" is two calls.
    """

    if at_least is None and at_most is None:
        raise ValueError("give at_least, at_most, or both")
    values = field(tokens, group, name)
    keep = tokens.mask
    if at_least is not None:
        keep = keep & (values >= at_least)
    if at_most is not None:
        keep = keep & (values <= at_most)
    return keep


def refine(tokens: TokenSet, mask: jax.Array) -> TokenSet:
    """Narrow a set to a subset mask, keeping it a set.

    Chaining stays cheap because nothing is compacted: the values array is
    untouched and only the mask shrinks.
    """

    return TokenSet(values=tokens.values, mask=tokens.mask & mask)


def count(tokens: TokenSet) -> jax.Array:
    """How many slots are occupied, per row."""

    return jnp.sum(tokens.mask.astype(jnp.int32), axis=-1)


def exists(tokens: TokenSet) -> jax.Array:
    """Whether any slot is occupied, per row."""

    return jnp.any(tokens.mask, axis=-1)


def masked_mean(tokens: TokenSet, group: str, name: str) -> jax.Array:
    """Mean of a named field over occupied slots only.

    Padded slots are excluded from both numerator and denominator, so an empty
    set yields 0 rather than a division by zero.
    """

    values = field(tokens, group, name)
    occupied = tokens.mask.astype(values.dtype)
    total = jnp.sum(values * occupied, axis=-1)
    return total / jnp.maximum(jnp.sum(occupied, axis=-1), 1.0)


def masked_sum(tokens: TokenSet, group: str, name: str) -> jax.Array:
    """Sum of a named field over occupied slots only."""

    values = field(tokens, group, name)
    return jnp.sum(values * tokens.mask.astype(values.dtype), axis=-1)


def field_of_view(
    tokens: TokenSet,
    group: str,
    *,
    forward: str,
    lateral: str,
    half_angle_degrees: float = 45.0,
) -> jax.Array:
    """Occupied slots inside a forward cone, as a mask.

    Both coordinates are agent-relative, so "in front of me" is a sign and
    ratio test rather than a transform.  Uses ``arctan2`` so it stays correct
    at the origin instead of dividing by a near-zero forward component.
    """

    if not 0.0 < half_angle_degrees <= 180.0:
        raise ValueError("half_angle_degrees must be in (0, 180]")
    ahead = field(tokens, group, forward)
    beside = field(tokens, group, lateral)
    bearing = jnp.abs(jnp.arctan2(beside, ahead))
    limit = jnp.deg2rad(half_angle_degrees)
    return tokens.mask & (bearing <= limit)


def has_field(group: str, name: str) -> bool:
    """Whether a group publishes a field, without raising.  Host-side, O(1).

    Useful for writing one query that adapts across groups whose schemas
    differ -- entities carry ``hostile``, terrain patches do not.
    """

    try:
        field_index(group, name)
    except KeyError:
        return False
    return True


__all__ = [
    "Selection",
    "count",
    "exists",
    "extreme",
    "field_of_view",
    "has_field",
    "masked_mean",
    "masked_sum",
    "matching",
    "nearest",
    "refine",
    "top_k",
]
