"""Branch rollouts from an arbitrary environment state.

Model-based agents need to ask "what happens from *here*", not only "what
happens from a reset".  ``BuiltAgent.step_factors`` already accepts any state,
so the missing pieces were a safe way to replicate one environment's state
across the batch and a rollout that starts from a state instead of a reset.

Two rules make this correct rather than merely plausible:

* **Only batch-owned leaves may be replicated.**  An exact Region state carries
  a shared immutable atlas whose leading dimension is a working-set capacity,
  not the environment batch.  Copying those leaves per row would corrupt the
  scene.  :func:`branch_from` applies the same test as
  ``select_environment_rows``: a leaf is batch-owned iff its leading axis
  equals the batch.
* **Imagination does not autoreset.**  A world model must see the real terminal
  boundary; silently restarting the episode would teach it that death is a
  teleport.  :func:`imagine` records the boundary and keeps stepping.

Stepping the real environment this way is *ground truth*, not a learned model.
It is the reference a learned dynamics model is trained against, and the way to
score that model's drift.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from adk.architecture.inputs import actor_policy_input
from adk.runtime.env_adapter import step_with_boundary
from adk.runtime.loop import LoopTransition


def batch_owned_leaves(state: Any, batch: int) -> tuple[str, ...]:
    """Return the paths this module would replicate for ``batch``.

    The complement is shared: provider atlases, trace-time constants, and
    anything whose leading axis is a capacity rather than the batch.
    """

    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    owned = []
    leaves, _ = jax.tree_util.tree_flatten_with_path(state)
    for path, leaf in leaves:
        shape = getattr(leaf, "shape", None)
        if shape and int(shape[0]) == batch:
            owned.append(jax.tree_util.keystr(path))
    return tuple(owned)


def branch_from(state: Any, *, row: int, batch: int) -> Any:
    """Replicate one environment's state across every row of the batch.

    The result is ``batch`` identical copies of environment ``row``, so a
    subsequent step with distinct keys explores ``batch`` divergent futures
    from the same real situation.  Shared immutable leaves are left untouched.
    """

    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    if isinstance(row, bool) or not isinstance(row, int):
        raise TypeError("row must be an integer")
    if not 0 <= row < batch:
        raise ValueError(f"row {row} is outside the batch of {batch}")

    def replicate(leaf):
        array = jnp.asarray(leaf)
        if array.ndim == 0 or array.shape[0] != batch:
            return leaf  # shared: capacity-shaped or scalar
        selected = array[row]
        return jnp.broadcast_to(selected[None, ...], array.shape)

    return jax.tree.map(replicate, state)


class ImaginedRollout(NamedTuple):
    """What an imagined branch produced.

    ``records`` is time-major: every leaf has a leading ``steps`` axis.
    """

    state: Any
    records: Any


def imagine(
    built: Any,
    state: Any,
    policy_input: Any,
    policy: Callable[..., Any],
    record: Callable[[Any], Any],
    steps: int,
    key: jax.Array,
    *,
    carry: Any = None,
) -> ImaginedRollout:
    """Roll forward from ``state`` without resetting, recording each step.

    ``state`` and ``policy_input`` are the pair ``reset`` and ``step`` both
    return; they must belong together.  The action mask cannot be recovered
    from a bare state -- ``DensePolicySurface.project`` needs the structured
    observation that produced it -- so the pair is passed rather than guessed.

    ``policy`` follows the ADK convention ``(carry, actor_input, key) ->
    (carry, factors)``; ``record`` receives the same :class:`LoopTransition`
    ``collect`` and ``compile_collector`` pass, so a recorder written for
    training works here unchanged.

    Unlike ``compile_collector`` this neither resets first nor autoresets on
    termination -- the boundary is reported and the episode is allowed to run
    past it, which is what a dynamics model needs to observe.
    """

    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be an integer")
    if steps < 1:
        raise ValueError("steps must be positive")

    batch = built.batch

    def body(loop, step_key):
        loop_state, loop_input, loop_carry = loop
        policy_key, step_split = jax.random.split(step_key)
        current_actor = actor_policy_input(loop_state, loop_input)
        next_carry, factors = policy(loop_carry, current_actor, policy_key)
        step = step_with_boundary(
            built,
            loop_state,
            factors,
            jax.random.split(step_split, batch),
        )
        transition = LoopTransition(
            state=loop_state,
            actor_input=current_actor,
            action_factors=factors,
            reward=step.reward,
            done=step.done,
            info=step.info,
            next_state=step.state,
            next_actor_input=actor_policy_input(step.state, step.policy_input),
            terminated=step.terminated,
            truncated=step.truncated,
        )
        return (
            step.state,
            step.policy_input,
            next_carry,
        ), record(transition)

    keys = jax.random.split(key, steps)
    (final_state, _, _), records = jax.lax.scan(
        body,
        (state, policy_input, carry),
        keys,
    )
    return ImaginedRollout(state=final_state, records=records)


__all__ = [
    "ImaginedRollout",
    "batch_owned_leaves",
    "branch_from",
    "imagine",
]
