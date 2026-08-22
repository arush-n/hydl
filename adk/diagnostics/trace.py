"""Composable, traced recorders for seeing what a rollout actually did.

A recorder runs *inside* the compiled scan, so anything it does is paid for on
every step of every environment.  Everything here is therefore array-only: no
reductions to Python, no device transfers, no branching on values.  The cost of
:func:`standard_trace` is a handful of bitwise ANDs and array references --
comparable to recording reward alone.

The pattern is composition rather than configuration:

    record = combine(my_record, standard_trace)

so a training recorder and a diagnostic one can be developed independently and
merged without either knowing about the other.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from adk.diagnostics.failures import any_failure, observation_failures


def boundary_trace(transition: Any) -> dict[str, jax.Array]:
    """Episode boundary, keeping terminated/truncated separate when known.

    A collapsed ``done`` cannot distinguish "died" from "ran out of time", and
    conflating them is a common source of wrong value targets.  When the split
    is unavailable this records ``done`` only rather than inventing a cause.
    """

    trace: dict[str, jax.Array] = {"done": transition.done}
    terminated = getattr(transition, "terminated", None)
    truncated = getattr(transition, "truncated", None)
    if terminated is not None:
        trace["terminated"] = terminated
    if truncated is not None:
        trace["truncated"] = truncated
    return trace


def action_trace(transition: Any) -> dict[str, jax.Array]:
    """What the policy chose and whether the environment accepted it.

    ``legal_action_count`` is the per-step size of the legal set, which is the
    cheapest early warning that a mask has collapsed: when it hits zero the
    policy is sampling from nothing.
    """

    actor = transition.actor_input
    trace: dict[str, jax.Array] = {
        "action_factors": transition.action_factors,
        "legal_action_count": jnp.sum(
            actor.action_mask.astype(jnp.int32), axis=-1
        ),
    }
    info = getattr(transition, "info", None)
    for name in ("action_valid", "action_surface_legal", "world_verb_legal"):
        value = getattr(info, name, None)
        if value is not None:
            trace[name] = value
    return trace


def failure_trace(transition: Any) -> dict[str, jax.Array]:
    """Every named failure flag, plus a single rolled-up gate."""

    legal = transition.actor_input.legal_observation
    trace = dict(observation_failures(legal))
    trace["any_failure"] = any_failure(legal)
    trace["observation_valid"] = legal.valid
    return trace


def standard_trace(transition: Any) -> dict[str, jax.Array]:
    """Reward, boundary, action legality, and failure flags in one recorder.

    The default diagnostic bundle.  Traced and sync-free, so it is safe to
    leave enabled during training rather than only when something breaks.
    """

    trace: dict[str, jax.Array] = {"reward": transition.reward}
    trace.update(boundary_trace(transition))
    trace.update(action_trace(transition))
    trace.update(failure_trace(transition))
    return trace


def combine(
    *recorders: Callable[[Any], dict[str, jax.Array]],
) -> Callable[[Any], dict[str, jax.Array]]:
    """Merge recorders into one, failing loudly on a key collision.

    A silent overwrite would drop whichever recorder ran first, which is
    exactly the sort of bug a diagnostic layer must not introduce.
    """

    if not recorders:
        raise ValueError("combine needs at least one recorder")

    def record(transition: Any) -> dict[str, jax.Array]:
        merged: dict[str, jax.Array] = {}
        for recorder in recorders:
            produced = recorder(transition)
            if not isinstance(produced, dict):
                raise TypeError(
                    "combine only merges recorders that return dicts; "
                    f"{getattr(recorder, '__name__', recorder)!r} returned "
                    f"{type(produced).__name__}"
                )
            clashing = sorted(set(produced) & set(merged))
            if clashing:
                raise ValueError(
                    f"recorders disagree on keys {clashing}; rename one side"
                )
            merged.update(produced)
        return merged

    return record


def summarize(records: Any) -> dict[str, jax.Array]:
    """Reduce a time-major trace to per-key scalars.  One pass, still traced.

    Boolean keys become rates, numeric keys become means, so a whole rollout
    collapses to something loggable without a host round-trip per key.
    """

    summary: dict[str, jax.Array] = {}
    for name, value in records.items():
        array = jnp.asarray(value)
        if array.dtype == jnp.bool_:
            summary[f"{name}.rate"] = jnp.mean(array.astype(jnp.float32))
        elif jnp.issubdtype(array.dtype, jnp.number):
            summary[f"{name}.mean"] = jnp.mean(array.astype(jnp.float32))
    return summary


__all__ = [
    "action_trace",
    "boundary_trace",
    "combine",
    "failure_trace",
    "standard_trace",
    "summarize",
]
