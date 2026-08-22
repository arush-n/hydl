"""Additive reward shaping, as a generic seam.

The environment exposes exactly four reward terms (`combat/types.py:182`), all
`CombatParams` floats. Reweighting them is a **task**. Anything those four
cannot express -- position, time, speed, accuracy, resource discipline -- needs
a value computed from state, and that is what this module carries.

A **minigame is one implementation of this**, not the thing itself. So is a
curriculum term, an agent-specific auxiliary reward, a one-off diagnostic, or a
penalty someone adds while chasing a bug. They all satisfy the same contract and
compose with each other, which is why this lives here rather than inside
`minigames/`.

## The contract

A shaping function is either::

    shape(next_state, info) -> (batch,) float                      # stateless
    shape(previous_state, next_state, info) -> (batch,) float      # a delta

The second form is marked with :func:`wants_previous` and invoked through
:func:`call`, which is the only thing that needs to know which it got. Both
states are the RAW environment state (the thing with ``.runtime``), not the
:class:`JaxEnvironmentState` wrapper that holds it on ``.environment``.

**Anything that pays for a change must take the second form.** Remembering the
previous tick in a Python dict works eagerly and is silently wrong under
``jax.jit``: the assignment runs once, at trace time, so repeated calls read a
dead trace's value and a ``lax.scan`` body -- traced once, run many times --
reads the same stale value on every iteration. Measured: a `displacement` term
that reports 3.0 then 4.0 eagerly reports 0.0, 0.0, 0.0, 0.0 inside a scan, and
raises nothing. Training compiles the step, so that is the only place it counts.
Receiving the previous state removes the memory rather than fixing it.

``observe_done(done)`` is still forwarded by :func:`compose` for terms that
carry their own state, but nothing in this repo needs it any more: an episode
boundary is readable from the state itself with :func:`terminated`, which is
pure and survives tracing.

## Deliberately additive

The native reward stays intact underneath. A shaped run is still comparable to
the unshaped baseline on the native terms alone, which is the only reason a
shaped number can be read at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import jax.numpy as jnp

#: The two calling contracts live in a leaf module so `adk.runtime` can apply
#: them without importing this package (which would be a cycle). Re-exported
#: here because this is where the seam is documented and where callers look.
from adk.contracts.shaping import WANTS_PREVIOUS, call, wants_previous

__all__ = ["Shaping", "compose", "apply", "resolve", "weighted",
           "wants_previous", "call", "WANTS_PREVIOUS"]


@dataclass(frozen=True)
class Shaping:
    """A named additive reward term computed from state.

    `teaches` is the behaviour it should produce; `tell` is the metric that
    would show it worked. A term whose `tell` cannot move on the scene it ran on
    is a broken setup rather than a failed agent -- state it, so the run is
    read as void instead of as a negative result.

    `requires` names scene properties the term is meaningless without
    (``"armed_opponent"``, ``"region"``, ...). It is checked by the caller
    rather than assumed, because a shaped term that silently reads a column the
    scene never populates pays a constant and looks like a working reward.
    """

    name: str
    #: ``(**kwargs) -> shape``. A factory rather than the function itself so a
    #: term can be parameterised per run and so stateful terms get a fresh
    #: closure instead of leaking the previous run's remembered tick.
    build: Callable[..., Callable]
    weight: float = 1.0
    requires: tuple[str, ...] = ()
    teaches: str = ""
    tell: str = ""
    #: Default kwargs handed to `build`; a caller's kwargs win over these.
    defaults: Mapping[str, Any] = field(default_factory=dict)

    def make(self, **kwargs) -> Callable:
        """Instantiate this term's shaping function."""

        return self.build(**{**dict(self.defaults), **kwargs})


def weighted(shape: Callable, weight: float) -> Callable:
    """Scale one shaping function, preserving its ``observe_done`` hook."""

    if weight == 1.0:
        return shape

    @wants_previous
    def scaled(previous_state, state, info):
        return jnp.float32(weight) * jnp.asarray(
            call(shape, previous_state, state, info), dtype=jnp.float32)

    _forward_done(scaled, [shape])
    return scaled


def compose(*terms: Callable) -> Callable | None:
    """Sum several shaping functions into one.

    Returns ``None`` for an empty list so a caller can tell "no shaping" from
    "shaping that happens to pay zero" -- those are different runs and only one
    of them is a baseline.
    """

    flat = [term for term in terms if term is not None]
    if not flat:
        return None
    if len(flat) == 1:
        return flat[0]

    @wants_previous
    def summed(previous_state, state, info):
        total = jnp.asarray(call(flat[0], previous_state, state, info),
                            dtype=jnp.float32)
        for term in flat[1:]:
            total = total + jnp.asarray(
                call(term, previous_state, state, info), dtype=jnp.float32)
        return total

    _forward_done(summed, flat)
    return summed


def _forward_done(wrapper: Callable, terms: Sequence[Callable]) -> None:
    """Give `wrapper` an ``observe_done`` that fans out to every stateful term.

    Without this, composing a stateful term with a stateless one would drop the
    boundary hook and the stateful term would pay a reset-sized reward on the
    first tick of every episode -- the exact failure `progress` guards against,
    reintroduced by the act of combining it with something else.
    """

    hooks = [getattr(term, "observe_done", None) for term in terms]
    hooks = [hook for hook in hooks if callable(hook)]
    if not hooks:
        return

    def observe_done(done):
        for hook in hooks:
            hook(done)

    wrapper.observe_done = observe_done      # type: ignore[attr-defined]


def apply(step, shape: Callable | None, weight: float = 1.0):
    """Wrap a ``step_detailed``-shaped callable to add `shape` to its reward.

    Kept signature-compatible with the `tasks.shaped` it generalises, so the
    existing `SceneConfig` path and every minigame keep working unchanged.
    """

    if shape is None:
        return step

    def wrapped(state, observation, actions, keys):
        next_state, next_observation, reward, done, mask, info = step(
            state, observation, actions, keys)
        # `state` is the previous raw state, which is exactly what a delta term
        # needs -- so this path supports both contracts with nothing extra.
        bonus = jnp.asarray(call(shape, state, next_state, info),
                            dtype=jnp.float32)
        return (next_state, next_observation, reward + weight * bonus,
                done, mask, info)

    return wrapped


def resolve(source: Any, *, registry: Mapping[str, Any] | None = None,
            **kwargs) -> Callable | None:
    """Turn whatever a caller passed into one shaping function.

    Accepts, in order of specificity:

    * ``None`` -- no shaping;
    * a plain callable ``(state, info) -> (batch,)``, used as-is;
    * a :class:`Shaping` (or anything with ``.make``/``.build`` and ``.weight``)
      -- instantiated and scaled;
    * a ``str`` -- looked up in `registry`, which is how minigames are named;
    * an iterable of any of the above -- composed.

    One entry point rather than four call sites that each accept a different
    subset, because "the console takes a name, the trainer takes a callable,
    the probe takes a Minigame" is how a seam ends up supporting three things
    and documenting one.
    """

    if source is None:
        return None

    if isinstance(source, str):
        # `is None`, not falsiness: an EMPTY registry is still a registry, and
        # conflating the two reports "no registry supplied" for a caller who
        # supplied one and simply has nothing registered yet. The second case
        # wants the name and the (empty) list of alternatives.
        if registry is None:
            raise ValueError(
                f"cannot resolve shaping {source!r} without a registry")
        if source not in registry:
            raise ValueError(
                f"unknown shaping {source!r}; have {sorted(registry)}")
        return resolve(registry[source], registry=registry, **kwargs)

    if callable(source) and not hasattr(source, "build"):
        return source

    if hasattr(source, "build"):
        weight = float(getattr(source, "weight", 1.0) or 1.0)
        make = getattr(source, "make", None)
        built = make(**kwargs) if callable(make) else source.build(**kwargs)
        return weighted(built, weight)

    if isinstance(source, Iterable):
        return compose(*[resolve(item, registry=registry, **kwargs)
                         for item in source])

    raise TypeError(
        f"cannot resolve {type(source).__name__} as reward shaping; pass a "
        "callable, a Shaping, a registry name, or a list of those")
