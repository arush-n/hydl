"""Which arguments a reward-shaping function takes, and how to call it.

Three lines of dispatch, in a leaf module, for one reason: both ends need it.
:mod:`adk.scenarios.shaping` defines the seam and re-exports these;
:mod:`adk.runtime.env_adapter` applies it inside ``step_factors``. Importing
the former from the latter is a cycle (``adk.scenarios`` pulls in probes ->
policy -> architecture -> env_adapter), and the alternative -- each side
deciding for itself which contract a term declares -- is precisely how the
console's carry-arity predicate drifted from its sibling wrapper and silently
mis-called every wrapped policy.

There are two contracts::

    shape(next_state, info) -> (batch,) float                   # stateless
    shape(previous_state, next_state, info) -> (batch,) float   # a delta

The second exists because a term that pays for a *change* must not remember the
previous tick itself. A Python-dict memory is correct eagerly and silently
wrong under ``jax.jit``: the store runs once, at trace time, so repeated calls
read a dead trace's value, and a ``lax.scan`` body -- traced once, run many
times -- reads the same stale value every iteration. Handing the term both
states removes the memory instead of guarding it.
"""

from __future__ import annotations

from typing import Callable

__all__ = ["WANTS_PREVIOUS", "wants_previous", "call"]

#: Attribute marking a shaping fn as taking ``(previous_state, next_state, info)``.
WANTS_PREVIOUS = "__shaping_wants_previous__"


def wants_previous(fn: Callable) -> Callable:
    """Mark `fn` as taking the previous state as its first argument.

    An explicit flag rather than `inspect.signature`, deliberately. A closure's
    arity is not a reliable statement about its contract -- defaults, ``*args``
    and wrappers all misreport it -- and a wrong arity guess calls the function
    with the wrong arguments rather than failing to call it. A flag is
    checkable, greppable, and survives every wrapper because :func:`weighted`
    and :func:`compose` re-apply it.
    """

    setattr(fn, WANTS_PREVIOUS, True)
    return fn


def call(shape: Callable, previous_state, next_state, info):
    """Invoke `shape` under whichever of the two contracts it declares."""

    if getattr(shape, WANTS_PREVIOUS, False):
        return shape(previous_state, next_state, info)
    return shape(next_state, info)
