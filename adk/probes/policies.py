"""Small driver policies whose purpose is to break the environment, not to win.

None of these learn. They exist to push the environment through states a
trained policy would reach only rarely, so that the invariants in
:mod:`adk.probes.invariants` get a chance to fire.

Each one satisfies the public ``Policy`` protocol -- ``(carry, actor_input,
key) -> (carry, int32[B, len(HEAD_SPANS)])`` -- so they drop straight into
``handle.compile_collector`` with no special runner.

Policies that need a step counter carry an ``initial_carry`` attribute; pass it
through, because ``scan`` requires the carry structure to be constant::

    probe = head_sweep("locomotion_gait_compass")
    handle.compile_collector(probe, record, 64, initial_carry=probe.initial_carry)
"""

from __future__ import annotations

from typing import Mapping, Sequence

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
)

from adk.policy.random_policy import uniform_legal
from adk.runtime.env_adapter import legal_mask, sample_actions


#: Head name -> (offset into the flat 99-wide mask, width).
HEAD_SPANS: Mapping[str, tuple[int, int]] = {}
_offset = 0
for _name, _size in zip(ARSENAL_POLICY_ACTION_HEAD_NAMES, ARSENAL_POLICY_ACTION_HEAD_SIZES):
    HEAD_SPANS[_name] = (_offset, _size)
    _offset += _size
del _offset, _name, _size

#: Total flat mask width, for logit construction.
ACTION_LOGIT_WIDTH: int = sum(ARSENAL_POLICY_ACTION_HEAD_SIZES)

#: How strongly a preference outweighs the uniform baseline.  Large enough to
#: dominate, finite so a masked-off preference still falls back to a legal
#: action instead of producing NaN.
PREFERENCE_STRENGTH = 30.0


def head_span(head: str) -> tuple[int, int]:
    """Offset and width of one action head in the flat mask.  O(1)."""

    try:
        return HEAD_SPANS[head]
    except KeyError as error:
        raise KeyError(
            f"unknown action head {head!r}; known: {sorted(HEAD_SPANS)}"
        ) from error


def _uniform_logits(mask: jax.Array) -> jax.Array:
    return jnp.zeros(mask.shape, dtype=jnp.float32)


def repeat(factors: jax.Array):
    """Emit the same action every step.

    Finds accumulation bugs: a cooldown that never clears, a resource that
    drains below zero, a status that stacks without bound.
    """

    held = jnp.asarray(factors)

    def policy(carry, _actor_input, _key):
        return carry, held

    return policy


def prefer(head: str, value: int, *, strength: float = PREFERENCE_STRENGTH):
    """Bias one head toward one value, sampling legally everywhere else.

    If the preferred value is masked off, the mask wins and a legal action is
    chosen instead -- so this never forces an illegal factor.  That fallback is
    itself informative: pair it with ``invariants`` to see whether the
    environment ever actually offers the value you asked for.
    """

    offset, size = head_span(head)
    if not 0 <= value < size:
        raise ValueError(f"{head!r} has {size} values; {value} is out of range")

    def policy(carry, actor_input, key):
        mask = legal_mask(actor_input)
        bias = jnp.zeros(ACTION_LOGIT_WIDTH, dtype=jnp.float32)
        bias = bias.at[offset + value].set(strength)
        logits = jnp.broadcast_to(bias, mask.shape)
        return carry, sample_actions(logits, mask, key)

    return policy


def prefer_periodic(head: str, value: int, period: int, *,
                    strength: float = PREFERENCE_STRENGTH):
    """``prefer`` on a duty cycle: request ``value`` every ``period`` steps.

    Spamming a stamina-gated head disables it. ``prefer`` asks for the option
    the instant it becomes affordable, which pins the pool at its floor and
    leaves the option illegal on almost every step -- measured, dodge direction
    3 was legal on **2.6%** of steps under ``prefer`` alone, below the 5% an
    execution control needs to mean anything, so the test it guarded was
    vacuous rather than failing.

    A duty cycle banks enough stamina for the option to actually be offered.
    Carry is the step index, so this is not a drop-in for ``prefer`` in
    collectors that put something else there.
    """

    offset, size = head_span(head)
    if not 0 <= value < size:
        raise ValueError(f"{head!r} has {size} values; {value} is out of range")
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    def policy(step, actor_input, key):
        mask = legal_mask(actor_input)
        choice = jnp.where(jnp.mod(step, period) == 0, value, 0)
        bias = (
            jax.nn.one_hot(offset + choice, ACTION_LOGIT_WIDTH, dtype=jnp.float32)
            * strength
        )
        return step + 1, sample_actions(jnp.broadcast_to(bias, mask.shape), mask, key)

    # Collectors seed the carry with `None` unless told otherwise, and `None`
    # is not a valid operand for `jnp.mod`. Pass this through as
    # `compile_collector(..., initial_carry=policy.initial_carry)`.
    policy.initial_carry = jnp.int32(0)
    return policy


def head_sweep(head: str, *, strength: float = PREFERENCE_STRENGTH):
    """Walk one head through every value in turn, one value per step.

    Systematic coverage of a single factor, which uniform sampling reaches only
    in proportion to head width.  Use it to answer "does value *k* of this head
    ever do anything?" -- a head value that is always legal and never changes
    the state is a dead action.

    Carries a step counter; pass ``initial_carry=policy.initial_carry``.
    """

    offset, size = head_span(head)

    def policy(step, actor_input, key):
        mask = legal_mask(actor_input)
        choice = jnp.mod(step, size)
        bias = (
            jax.nn.one_hot(offset + choice, ACTION_LOGIT_WIDTH, dtype=jnp.float32)
            * strength
        )
        logits = jnp.broadcast_to(bias, mask.shape)
        return step + 1, sample_actions(logits, mask, key)

    policy.initial_carry = jnp.int32(0)
    return policy


def head_random(head: str, *, strength: float = PREFERENCE_STRENGTH):
    """Randomize one head hard, leave the rest uniform-legal.

    Isolates blame.  When a full uniform-legal run trips an invariant, rerun
    with one head at a time to find which factor causes it.
    """

    offset, size = head_span(head)

    def policy(carry, actor_input, key):
        mask = legal_mask(actor_input)
        choice_key, sample_key = jax.random.split(key)
        batch = mask.shape[0]
        choice = jax.random.randint(choice_key, (batch,), 0, size)
        bias = (
            jax.nn.one_hot(offset + choice, ACTION_LOGIT_WIDTH, dtype=jnp.float32)
            * strength
        )
        return carry, sample_actions(bias, mask, sample_key)

    return policy


def sequence(factors: Sequence[jax.Array]):
    """Replay a fixed action sequence, cycling when it runs out.

    The reproduction tool: once a random probe trips something, record the
    factors it played and replay them deterministically.

    Carries a step counter; pass ``initial_carry=policy.initial_carry``.
    """

    if not factors:
        raise ValueError("sequence needs at least one action")
    stacked = jnp.stack([jnp.asarray(f) for f in factors])

    def policy(step, _actor_input, _key):
        return step + 1, stacked[jnp.mod(step, stacked.shape[0])]

    policy.initial_carry = jnp.int32(0)
    return policy


__all__ = [
    "ACTION_LOGIT_WIDTH",
    "HEAD_SPANS",
    "PREFERENCE_STRENGTH",
    "head_random",
    "head_span",
    "head_sweep",
    "prefer",
    "repeat",
    "sequence",
    "uniform_legal",
]
