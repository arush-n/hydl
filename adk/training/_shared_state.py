"""Non-mutating Gym PPO binding for shared provider-state leaves."""

from __future__ import annotations

import inspect
from types import FunctionType
from typing import Any, Callable

from hytalegym.jax.training import ppo as gym_ppo

from adk.runtime.tree import select_environment_rows


_PRIVATE_SIGNATURES = {
    "_collect_rollout": (
        "train_state",
        "key",
        "config",
        "environment",
    ),
    "_rollout_phase": (
        "train_state",
        "key",
        "config",
        "environment",
    ),
}


def _verify_gym_ppo_surface() -> None:
    for name, expected in _PRIVATE_SIGNATURES.items():
        function = getattr(gym_ppo, name, None)
        if function is None:
            raise RuntimeError(f"Gym PPO private port disappeared: {name}")
        actual = tuple(inspect.signature(function).parameters)
        if actual != expected:
            raise RuntimeError(
                f"Gym PPO private port changed: {name} expected "
                f"{expected!r}, found {actual!r}"
            )


_verify_gym_ppo_surface()


def _rebind(function: Callable[..., Any], **bindings: Any):
    """Clone one function with isolated globals and unchanged bytecode."""

    namespace = dict(function.__globals__)
    namespace.update(bindings)
    rebound = FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    rebound.__kwdefaults__ = function.__kwdefaults__
    rebound.__annotations__ = dict(function.__annotations__)
    rebound.__doc__ = function.__doc__
    rebound.__module__ = function.__module__
    rebound.__qualname__ = function.__qualname__
    return rebound


def make_shared_state_train_step(
    config,
    *,
    environment,
    compile: bool = True,
):
    """Build live Gym PPO with only its tree-selector dependency replaced."""

    collect_rollout = _rebind(
        gym_ppo._collect_rollout,
        _batch_select=select_environment_rows,
    )
    rollout_phase = _rebind(
        gym_ppo._rollout_phase,
        _collect_rollout=collect_rollout,
    )
    make_train_step = _rebind(
        gym_ppo.make_train_step,
        _rollout_phase=rollout_phase,
    )
    return make_train_step(
        config,
        environment=environment,
        compile=compile,
    )


__all__ = ["make_shared_state_train_step"]
