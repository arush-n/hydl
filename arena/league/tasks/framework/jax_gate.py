"""Compile a goal against the narrow Arena/JAX scoring contract."""

from __future__ import annotations

import copy
import json
import threading
import time
from collections import OrderedDict
from hashlib import sha256
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib

from arena.jax_contract import GROUP_FEATURES, READABLE_GROUPS
from arena.tasks.framework.criteria import criterion_groups
from arena.tasks.framework.goals import Goal
from arena.tasks.framework.rewards import TaskReward

SCHEMA = "arena_jax_goal_compile_v2"
REWARD_SCHEMA = "arena_jax_reward_compile_v3"
PROBE_BATCH = 2
PROBE_TICKS = 7
_CACHE_LIMIT = 16
_CACHE: OrderedDict[str, tuple[Any, dict[str, Any]]] = OrderedDict()
_REWARD_CACHE: OrderedDict[str, tuple[Any, dict[str, Any]]] = OrderedDict()
_LOCK = threading.Lock()


def _dtype(group: str):
    if group.endswith("_i32"):
        return jnp.int32
    if group.endswith("_mask"):
        return jnp.bool_
    return jnp.float32


def _callable_digest(function: Any) -> str:
    code = getattr(function, "__code__", None)
    payload = getattr(function, "__name__", type(function).__name__).encode()
    if code is not None:
        payload += code.co_code + repr(code.co_consts).encode()
    return sha256(payload).hexdigest()


def compile_goal_contract(
    goal: Goal, *, max_groups: int = 1
) -> tuple[dict[str, Any], bool]:
    """Compile and execute ``goal``; return its receipt and cache-hit flag.

    The executable is retained in a bounded process cache. Full rollout
    compilation is intentionally separate because it needs a captured world.
    """

    groups = criterion_groups(goal)
    if groups is None:
        raise ValueError("custom goals must declare static observation groups")
    unknown = sorted(groups - READABLE_GROUPS)
    if unknown:
        raise ValueError(
            "custom goal uses unreadable observation groups: " + ", ".join(unknown)
        )
    if not groups:
        raise ValueError("custom goal must read at least one observation group")
    if max_groups < 1 or len(groups) > max_groups:
        raise ValueError(
            f"custom goal needs {len(groups)} observation groups; optimized "
            f"creator limit is {max_groups}"
        )

    ordered = tuple(sorted(groups))
    ticks = max(PROBE_TICKS, goal.minimum_ticks)
    contract = {
        "goal": goal.describe(),
        "implementation": _callable_digest(goal.criterion),
        "required_groups": {group: list(GROUP_FEATURES[group]) for group in ordered},
        "probe_batch": PROBE_BATCH,
        "probe_ticks": ticks,
    }
    digest = sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    backend = jax.default_backend()
    device = jax.devices(backend)[0]
    cache_key = ":".join(
        (digest, jax.__version__, jaxlib.__version__, backend, device.device_kind)
    )

    with _LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            _CACHE.move_to_end(cache_key)
            return copy.deepcopy(cached[1]), True

        record = {
            group: jnp.zeros(
                (ticks, PROBE_BATCH, len(GROUP_FEATURES[group])),
                dtype=_dtype(group),
            )
            for group in ordered
        }

        # A plain function is required here: slotted Goal instances cannot be
        # weak-referenced by jax.jit's callable cache.
        def evaluate(values):
            return goal(values)

        started = time.perf_counter()
        try:
            executable = jax.jit(evaluate).lower(record).compile()
            compiled_at = time.perf_counter()
            output = executable(record)
            jax.block_until_ready(output)
        except Exception as exc:
            raise ValueError(
                f"goal {goal.label!r} failed JAX compilation: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finished = time.perf_counter()
        if output.shape != (PROBE_BATCH,) or output.dtype != jnp.bool_:
            raise ValueError(
                f"goal {goal.label!r} compiled to {output.dtype}{output.shape}; "
                f"expected bool[{PROBE_BATCH}]"
            )

        receipt = {
            "schema": SCHEMA,
            "criterion_compiled": True,
            "contract_digest": digest,
            "backend": backend,
            "device": str(device),
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "required_groups": list(ordered),
            "narrow_recording": True,
            "fused_evaluator_eligible": True,
            "minimum_ticks": goal.minimum_ticks,
            "inputs": {
                group: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                for group, value in record.items()
            },
            "output": {"shape": list(output.shape), "dtype": str(output.dtype)},
            "compile_ms": round((compiled_at - started) * 1_000, 3),
            "execute_ms": round((finished - compiled_at) * 1_000, 3),
            "full_rollout_compile": "pending_region_capture",
        }
        _CACHE[cache_key] = (executable, receipt)
        if len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
        return copy.deepcopy(receipt), False


def compile_reward_contract(
    reward: TaskReward, *, max_groups: int = 1
) -> tuple[dict[str, Any], bool]:
    """Compile one transition reward and return a device-specific receipt."""

    if not isinstance(reward, TaskReward):
        raise TypeError("reward must be a TaskReward")
    groups = tuple(sorted(reward.required_groups))
    if len(groups) > max_groups:
        raise ValueError(
            f"custom reward needs {len(groups)} observation groups; optimized "
            f"creator limit is {max_groups}"
        )
    contract = {
        "reward": reward.describe(),
        "implementations": [
            _callable_digest(signal.evaluate) for signal in reward.signals
        ],
        "required_groups": {group: list(GROUP_FEATURES[group]) for group in groups},
        "probe_batch": PROBE_BATCH,
    }
    digest = sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    backend = jax.default_backend()
    device = jax.devices(backend)[0]
    cache_key = ":".join(
        (digest, jax.__version__, jaxlib.__version__, backend, device.device_kind)
    )

    with _LOCK:
        cached = _REWARD_CACHE.get(cache_key)
        if cached is not None:
            _REWARD_CACHE.move_to_end(cache_key)
            return copy.deepcopy(cached[1]), True

        previous = {
            group: jnp.zeros(
                (PROBE_BATCH, len(GROUP_FEATURES[group])), dtype=_dtype(group)
            )
            for group in groups
        }
        current = {
            group: jnp.ones(
                (PROBE_BATCH, len(GROUP_FEATURES[group])), dtype=_dtype(group)
            )
            for group in groups
        }
        native = jnp.zeros((PROBE_BATCH,), dtype=jnp.float32)
        terminal = jnp.zeros((PROBE_BATCH,), dtype=jnp.bool_)

        evidence_names = tuple(sorted(reward.required_completion_evidence))
        victory = jnp.arange(PROBE_BATCH) == 1

        def evaluate(base_reward, before, after, done, exact_victory):
            return reward.apply(
                base_reward,
                before,
                after,
                terminal=done,
                completion_evidence={name: exact_victory for name in evidence_names},
            )

        started = time.perf_counter()
        try:
            executable = (
                jax.jit(evaluate)
                .lower(native, previous, current, terminal, victory)
                .compile()
            )
            compiled_at = time.perf_counter()
            output = executable(native, previous, current, terminal, victory)
            jax.block_until_ready(output)
        except Exception as exc:
            raise ValueError(
                f"reward {reward.label!r} failed JAX compilation: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finished = time.perf_counter()
        if output.shape != (PROBE_BATCH,) or output.dtype != jnp.float32:
            raise ValueError(
                f"reward {reward.label!r} compiled to {output.dtype}{output.shape}; "
                f"expected float32[{PROBE_BATCH}]"
            )
        if not bool(jnp.all(jnp.isfinite(output))):
            raise ValueError(f"reward {reward.label!r} produced non-finite values")

        receipt = {
            "schema": REWARD_SCHEMA,
            "reward_compiled": True,
            "contract_digest": digest,
            "backend": backend,
            "device": str(device),
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "required_groups": list(groups),
            "terminal_aware": True,
            "completion_evidence_aware": True,
            "completion_evidence": list(evidence_names),
            "output": {"shape": list(output.shape), "dtype": str(output.dtype)},
            "compile_ms": round((compiled_at - started) * 1_000, 3),
            "execute_ms": round((finished - compiled_at) * 1_000, 3),
            "fused_environment_eligible": True,
            "full_rollout_compile": "pending_region_capture",
        }
        _REWARD_CACHE[cache_key] = (executable, receipt)
        if len(_REWARD_CACHE) > _CACHE_LIMIT:
            _REWARD_CACHE.popitem(last=False)
        return copy.deepcopy(receipt), False


__all__ = [
    "REWARD_SCHEMA",
    "SCHEMA",
    "compile_goal_contract",
    "compile_reward_contract",
]
