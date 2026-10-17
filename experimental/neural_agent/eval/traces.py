"""Trace dataset for the learned J1 instance, written from a hierarchical run.

:func:`record_j0_traces` runs a compiled hierarchy and writes one ``.npz`` file.
Rows are time-major: row ``t * B + b`` is lane ``b`` at tick ``t``. With
N = T * B, the arrays are:

* ``features``: float32[N, 10]. ``SharedBeliefState.features`` that the manager saw.
* ``feature_known``: bool[N, 10]. ``SharedBeliefState.feature_known``.
* ``latent``: float32[N, 64]. ``SharedBeliefState.latent``.
* ``exec_skill_index``: int32[N, 4]. Executive slot skill indices, -1 when empty.
* ``exec_priority``: float32[N, 4].
* ``exec_params``: float32[N, 4, 4], zero-padded.
* ``exec_goal_valid``: bool[N, 4].
* ``exec_budget``: float32[N, 3] = [jev_latency_ms, search_expansions, executive_calls_per_minute].
* ``exec_version``: int32[N].
* ``current_skill``: int32[N]. The skill in force before this tick's decision.
* ``target_skill``: int32[N]. The manager's proposed skill where the decision
  was due, otherwise ``current_skill``.
* ``target_replace``: bool[N]. True where the decision replaced the goal
  (``HierarchicalStep.goal_switched``).
* ``manager_due``: bool[N]. True where the manager's decision was due this tick
  (``HierarchicalStep.manager_due``). On the other rows ``target_skill`` is a copy
  of ``current_skill``, so a behaviour-cloning reader keeps the due rows only
  (added 2026-10-17; ``eval/record_traces.py`` is that reader).

The executive is fixed for a whole run, so its lowered arrays repeat on every
row. The layout follows the composition brief. ``jev/train_j1_ppo.py`` trains
on a synthetic environment and reads no trace file, so this layout has no
reader in the repository to check against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from adk.architecture.inputs import actor_policy_input
from adk.runtime.hierarchical import (
    HierarchicalActorInputFn,
    HierarchicalContextFn,
    HierarchicalLoopTransition,
    compile_hierarchical_collector,
    jax_hierarchical_context,
)
from experimental.neural_agent.contracts import (
    BELIEF_FEATURE_COUNT,
    BELIEF_LATENT_WIDTH,
    MAX_GOALS,
    PARAM_WIDTH,
    ExecutiveArrays,
    ExecutiveState,
    executive_arrays,
)

#: Per-tick record keys, each shaped [T, B, ...] after the collector stacks them.
TICK_RECORD_KEYS: tuple[str, ...] = (
    "features",
    "feature_known",
    "latent",
    "current_skill",
    "target_skill",
    "target_replace",
    "manager_due",
)

#: Every key of the trace file, in the order it is written.
TRACE_KEYS: tuple[str, ...] = (
    "features",
    "feature_known",
    "latent",
    "exec_skill_index",
    "exec_priority",
    "exec_params",
    "exec_goal_valid",
    "exec_budget",
    "exec_version",
    "current_skill",
    "target_skill",
    "target_replace",
    "manager_due",
)

#: Required dtype of every trace key, so that a reader can rely on it.
_TRACE_DTYPES: dict[str, np.dtype] = {
    "features": np.dtype(np.float32),
    "feature_known": np.dtype(np.bool_),
    "latent": np.dtype(np.float32),
    "exec_skill_index": np.dtype(np.int32),
    "exec_priority": np.dtype(np.float32),
    "exec_params": np.dtype(np.float32),
    "exec_goal_valid": np.dtype(np.bool_),
    "exec_budget": np.dtype(np.float32),
    "exec_version": np.dtype(np.int32),
    "current_skill": np.dtype(np.int32),
    "target_skill": np.dtype(np.int32),
    "target_replace": np.dtype(np.bool_),
    "manager_due": np.dtype(np.bool_),
}


def trace_record(transition: HierarchicalLoopTransition) -> dict[str, jax.Array]:
    """Return the per-tick trace arrays for one collector step, shaped [B, ...].

    This is the ``record_fn`` of :func:`record_j0_traces`. The belief is the one
    the manager saw this tick. ``current_skill`` is the goal in force before the
    decision. ``target_skill`` is the manager's proposal where the decision was
    due and the current skill otherwise. ``target_replace`` is ``goal_switched``.
    ``manager_due`` is ``step.manager_due``.

    Args:
        transition: The HierarchicalLoopTransition from the collector.

    Returns:
        A dict with the keys in :data:`TICK_RECORD_KEYS`.
    """

    step = transition.hierarchy_step
    belief = step.belief_update.belief
    current = transition.hierarchy_state.carry.goal.payload.skill_index
    proposed = step.manager_proposal.goal.payload.skill_index
    return {
        "features": jnp.asarray(belief.features, dtype=jnp.float32),
        "feature_known": jnp.asarray(belief.feature_known, dtype=jnp.bool_),
        "latent": jnp.asarray(belief.latent, dtype=jnp.float32),
        "current_skill": jnp.asarray(current, dtype=jnp.int32),
        "target_skill": jnp.where(step.manager_due, proposed, current).astype(jnp.int32),
        "target_replace": jnp.asarray(step.goal_switched, dtype=jnp.bool_),
        "manager_due": jnp.asarray(step.manager_due, dtype=jnp.bool_),
    }


def flatten_trace(
    records: Mapping[str, Any],
    executive: ExecutiveArrays,
) -> dict[str, np.ndarray]:
    """Flatten [T, B, ...] tick records and the lowered executive into N = T * B rows.

    Args:
        records: The keys in :data:`TICK_RECORD_KEYS`. Each has leading shape [T, B].
        executive: The ExecutiveArrays for the run, with leading shape [B].

    Returns:
        A dict with every key in :data:`TRACE_KEYS`. Each value has leading shape [N].

    Raises:
        ValueError: If a key is missing, or the shapes disagree with [T, B] or [B].
    """

    missing = [key for key in TICK_RECORD_KEYS if key not in records]
    if missing:
        raise ValueError(f"tick records are missing keys {missing}")
    host = {key: np.asarray(records[key]) for key in TICK_RECORD_KEYS}
    ticks, lanes = host["current_skill"].shape[:2]
    rows = ticks * lanes
    out: dict[str, np.ndarray] = {}
    for key in TICK_RECORD_KEYS:
        value = host[key]
        if value.shape[:2] != (ticks, lanes):
            raise ValueError(f"{key} must have leading shape [T, B] = {(ticks, lanes)}, got {value.shape}")
        out[key] = value.reshape((rows,) + value.shape[2:])
    executive_leaves = {
        "exec_skill_index": executive.skill_index,
        "exec_priority": executive.priority,
        "exec_params": executive.params,
        "exec_goal_valid": executive.goal_valid,
        "exec_budget": executive.budget,
        "exec_version": executive.version,
    }
    for key, leaf in executive_leaves.items():
        array = np.asarray(leaf)
        if array.shape[:1] != (lanes,):
            raise ValueError(f"{key} must have leading shape [B] = {(lanes,)}, got {array.shape}")
        tiled = np.broadcast_to(array[None], (ticks,) + array.shape)
        out[key] = tiled.reshape((rows,) + array.shape[1:])
    return {key: out[key] for key in TRACE_KEYS}


def write_trace(path: str | Path, arrays: Mapping[str, np.ndarray]) -> Path:
    """Validate a flattened trace and write it as an ``.npz`` file.

    Args:
        path: Output path. It must end in ``.npz``, so that ``np.savez`` writes
            exactly this name.
        arrays: The keys in :data:`TRACE_KEYS`, with the dtypes and shapes of
            the module docstring. N is the leading length of ``current_skill``.

    Returns:
        The path written, as a :class:`pathlib.Path`.

    Raises:
        ValueError: If the suffix is not ``.npz``, a key is missing or unexpected,
            or a dtype or shape is wrong.
    """

    target = Path(path)
    if target.suffix != ".npz":
        raise ValueError(f"trace path must end in .npz, got {target.name!r}")
    keys = set(arrays)
    if keys != set(TRACE_KEYS):
        raise ValueError(
            f"trace keys must be exactly {TRACE_KEYS}; missing {sorted(set(TRACE_KEYS) - keys)}, "
            f"unexpected {sorted(keys - set(TRACE_KEYS))}"
        )
    rows = int(np.asarray(arrays["current_skill"]).shape[0])
    expected_shapes: dict[str, tuple[int, ...]] = {
        "features": (rows, BELIEF_FEATURE_COUNT),
        "feature_known": (rows, BELIEF_FEATURE_COUNT),
        "latent": (rows, BELIEF_LATENT_WIDTH),
        "exec_skill_index": (rows, MAX_GOALS),
        "exec_priority": (rows, MAX_GOALS),
        "exec_params": (rows, MAX_GOALS, PARAM_WIDTH),
        "exec_goal_valid": (rows, MAX_GOALS),
        "exec_budget": (rows, 3),
        "exec_version": (rows,),
        "current_skill": (rows,),
        "target_skill": (rows,),
        "target_replace": (rows,),
        "manager_due": (rows,),
    }
    for key in TRACE_KEYS:
        value = np.asarray(arrays[key])
        if value.dtype != _TRACE_DTYPES[key]:
            raise ValueError(f"{key} must have dtype {_TRACE_DTYPES[key]}, got {value.dtype}")
        if value.shape != expected_shapes[key]:
            raise ValueError(f"{key} must have shape {expected_shapes[key]}, got {value.shape}")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **{key: np.asarray(arrays[key]) for key in TRACE_KEYS})
    return target


def record_j0_traces(
    path: str | Path,
    *,
    built: Any,
    core: Any,
    initial_state: Any,
    executive: ExecutiveState,
    skill_ids: tuple[str, ...] | list[str],
    ticks: int,
    seed: int,
    context_fn: HierarchicalContextFn = jax_hierarchical_context,
    actor_input_fn: HierarchicalActorInputFn = actor_policy_input,
) -> Path:
    """Run one hierarchical episode batch and write its J1 trace dataset.

    The run uses one compiled collector and the key ``fold_in(key(seed), 0)``.
    The records are flattened with :func:`flatten_trace`, then written with
    :func:`write_trace`.

    Args:
        path: Output ``.npz`` path.
        built: A structured environment with ``batch``, ``reset`` and a step
            method, as ``adk.runtime.hierarchical`` requires.
        core: A ``HierarchicalAgentCore`` whose manager is the instance to record.
        initial_state: The reset ``HierarchicalCoreState`` for ``built.batch`` lanes.
        executive: The executive state E_t that the manager is bound to.
        skill_ids: The catalogue in dispatcher order. The executive's goals must name ids in it.
        ticks: Ticks per run, a positive int.
        seed: Root seed for the run, a non-negative int.
        context_fn: Maps ``(info, candidate_actor_input, previous_actor_input)`` to a DecisionContext.
        actor_input_fn: Maps ``(environment_state, policy_input)`` to an ActorPolicyInput.

    Returns:
        The path written.

    Raises:
        ValueError: If ``seed`` is negative or not an integer. Also from :func:`flatten_trace`
            or :func:`write_trace`, or from ``executive_arrays`` when a goal names an id that is
            not in ``skill_ids``.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    batch = int(built.batch)
    collector = compile_hierarchical_collector(
        built,
        core,
        trace_record,
        ticks,
        initial_state=initial_state,
        context_fn=context_fn,
        actor_input_fn=actor_input_fn,
    )
    _, records = collector(jax.random.fold_in(jax.random.key(seed), 0))
    jax.block_until_ready(records)
    host = {key: np.asarray(value) for key, value in records.items()}
    arrays = flatten_trace(host, executive_arrays(executive, tuple(skill_ids), batch))
    return write_trace(path, arrays)


__all__ = [
    "TICK_RECORD_KEYS",
    "TRACE_KEYS",
    "flatten_trace",
    "record_j0_traces",
    "trace_record",
    "write_trace",
]
