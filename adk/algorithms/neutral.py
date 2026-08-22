"""Algorithm-neutral gradient updates over ADK-collected data.

``handle.ppo`` is one consumer of the SDK, not the SDK's training story.  The
compiled collector is already algorithm-neutral -- a caller chooses what each
transition records -- but there was no neutral way to turn those records into a
parameter update, so behaviour cloning, DAgger, and auxiliary-loss agents had to
either import the PPO module or hand-roll the scan.

This closes that gap with the smallest surface that is still general: the caller
owns the parameters, the policy they induce, what gets recorded, the loss, and
the optimiser.  The ADK owns collection and the compiled update.

The optimiser is duck-typed on ``init``/``update`` -- the optax interface --
rather than imported, so the ADK gains no new dependency and any optax
transformation works unchanged.
"""

from __future__ import annotations

from typing import Any, Callable

import jax


def apply_updates(parameters: Any, updates: Any) -> Any:
    """Add optimiser updates to parameters leafwise."""

    return jax.tree_util.tree_map(
        lambda parameter, update: parameter + update,
        parameters,
        updates,
    )


#: Rollout length at which episodes actually complete in the Arsenal scenes.
#:
#: Measured 2026-07-31: episodes run roughly 206-300 steps, so at the common
#: smoke-test default of 16 no episode ever finishes and every episode-level
#: metric silently reads as zero.  Use this for any run whose results depend on
#: completed episodes.
DEFAULT_ROLLOUT_STEPS = 256


def run_updates(
    update: Callable[[Any, Any, jax.Array], tuple[Any, Any, Any]],
    parameters: Any,
    optimizer_state: Any,
    key: jax.Array,
    *,
    updates: int,
    on_update: Callable[[int, Any, Any], None] | None = None,
) -> tuple[Any, Any, list[Any]]:
    """Run ``updates`` compiled steps, threading keys and optimiser state.

    Deliberately a host-side loop: ``on_update(index, value, parameters)`` is
    where logging, evaluation, and checkpointing belong, and those are Python
    side effects that cannot live inside a compiled scan.  The expensive part is
    already compiled inside ``update``.
    """

    if isinstance(updates, bool) or not isinstance(updates, int):
        raise TypeError("updates must be an integer")
    if updates < 1:
        raise ValueError("updates must be positive")
    if on_update is not None and not callable(on_update):
        raise TypeError("on_update must be callable")

    history: list[Any] = []
    for index in range(updates):
        key, step_key = jax.random.split(key)
        parameters, optimizer_state, value = update(
            parameters,
            optimizer_state,
            step_key,
        )
        history.append(value)
        if on_update is not None:
            on_update(index, value, parameters)
    return parameters, optimizer_state, history


def make_update(
    handle: Any,
    *,
    policy_of: Callable[[Any], Any],
    record: Callable[..., Any],
    loss: Callable[..., Any],
    optimizer: Any,
    steps: int,
    initial_carry: Any = None,
    has_aux: bool = False,
    compile: bool = True,
) -> Callable[[Any, Any, jax.Array], tuple[Any, Any, Any]]:
    """Compile one collect-then-descend update.

    ``policy_of`` turns the current parameters into a :class:`Policy`, so the
    data is always collected on-policy for the parameters being updated.  Pass a
    constant-returning ``policy_of`` for off-policy or purely offline losses.

    ``loss`` receives ``(parameters, batch)`` and returns a scalar, or
    ``(scalar, aux)`` when ``has_aux``.

    Returns ``update(parameters, optimizer_state, key)`` ->
    ``(parameters, optimizer_state, aux)`` where ``aux`` is the loss value, or
    ``(loss, aux)`` when ``has_aux``.
    """

    if not callable(policy_of):
        raise TypeError("policy_of must be callable: parameters -> Policy")
    if not callable(record):
        raise TypeError("record must be callable")
    if not callable(loss):
        raise TypeError("loss must be callable: (parameters, batch) -> scalar")
    for method in ("init", "update"):
        if not callable(getattr(optimizer, method, None)):
            raise TypeError(
                "optimizer must expose optax-style "
                f"{method!r}; got {type(optimizer).__name__}"
            )
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be an integer")
    if steps < 1:
        raise ValueError("steps must be positive")

    def update(parameters, optimizer_state, key):
        _state, batch = handle.collect(
            policy_of(parameters),
            record,
            key,
            steps,
            initial_carry=initial_carry,
        )
        value, gradients = jax.value_and_grad(loss, has_aux=has_aux)(
            parameters,
            batch,
        )
        updates, optimizer_state = optimizer.update(
            gradients,
            optimizer_state,
            parameters,
        )
        return apply_updates(parameters, updates), optimizer_state, value

    return jax.jit(update) if compile else update
