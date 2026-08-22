"""Batched rollout under lax.scan with masked autoreset.

Training is rollout-bound, not optimizer-bound (66.86% geometry / 4.32%
optimizer), so this loop is where throughput is won or lost. No host sync
inside the scan.

Terminal handling follows the gym's own PPO loop: step, reset
unconditionally, then select per row. Both branches are computed every step
-- that is deliberate. ``lax.cond`` on a traced per-row value does not
compile under batching, and a Python branch would not compile at all.

The policy carries state so a recurrent belief (PLAN.md layer 3) and a GRU
actor (layer 4) are expressible. **The carry resets on terminal alongside
the environment** -- a belief that survives an episode boundary is a silent
training corruption, not a visible failure.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from adk.development import EnvironmentDiagnostics
from adk.policy import Policy
from adk.runtime.env_adapter import BuiltAgent
from adk.runtime.episode import EpisodeBoundary
from adk.runtime.loop import LoopTransition, collect


class Trajectory(NamedTuple):
    """(T, B) rollout output.

    ``observations`` and ``action_masks`` are the ones the policy *acted on*,
    captured before the step -- PPO recomputes log-probs and values against
    them, and a recorded mask is the only way to reproduce the categorical
    that produced ``actions``. Without both, this is a data-collection loop
    and cannot train.
    """

    observations: jax.Array  # float32 (T, B, observation_size), pre-step
    action_masks: jax.Array  # bool (T, B, logit_width), pre-step
    actions: jax.Array  # int32 (T, B, heads) per-head factors
    rewards: jax.Array  # float32
    done: jax.Array  # bool, terminal BEFORE the autoreset
    info: Any  # ArsenalEnvironmentInfo, retained for training/offline events
    terminated: jax.Array | None = None
    truncated: jax.Array | None = None

    @property
    def boundary(self) -> EpisodeBoundary:
        """Time-major boundary values with explicit cause availability."""

        return EpisodeBoundary(self.done, self.terminated, self.truncated)

    @property
    def diagnostics(self) -> EnvironmentDiagnostics:
        """Name common fields while preserving the complete time-major info."""

        return EnvironmentDiagnostics.from_info(self.info)


def rollout(
    built: BuiltAgent,
    policy_fn: Policy,
    key: jax.Array,
    steps: int,
    initial_carry: Any = None,
) -> tuple[Any, Trajectory]:
    """Run ``steps`` batched transitions with autoreset. ``steps`` is static.

    ``policy_fn(carry, actor_input, key) -> (carry, action_factors)``.
    ``actor_input`` retains both the dense bridge row and Gym's structured
    legal learner observation. A stateless policy threads ``None`` and is
    unaffected.
    """

    def record(transition: LoopTransition) -> Trajectory:
        return Trajectory(
            observations=transition.actor_input.observation,
            action_masks=transition.actor_input.action_mask,
            actions=transition.action_factors,
            rewards=jnp.asarray(transition.reward, dtype=jnp.float32),
            done=jnp.asarray(transition.done, dtype=jnp.bool_),
            info=transition.info,
            terminated=transition.terminated,
            truncated=transition.truncated,
        )

    return collect(
        built,
        policy_fn,
        record,
        key,
        steps,
        initial_carry=initial_carry,
    )
