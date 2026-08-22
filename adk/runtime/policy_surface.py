"""Reusable projections from the structured JAX environment to policy tensors.

The structured Arsenal environment is the SDK's canonical runtime.  Dense
observations and masks are useful ports for neural policies and the native
bridge, but they are projections of that runtime rather than the runtime
contract itself.  Keeping the projection here gives every algorithm the same
bit-exact path without making PPO a dependency of environment construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax

from hytalegym.jax.combat.environment import EnvironmentSpec
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_SCHEMA,
    arsenal_policy_observation,
)


class PolicyInput(NamedTuple):
    """Exact dense actor projection shared by JAX and native bridge ports."""

    observation: jax.Array
    action_mask: jax.Array

    @property
    def dense_observation(self) -> jax.Array:
        """Make the projected nature of ``observation`` explicit."""

        return self.observation


@dataclass(frozen=True, slots=True)
class DensePolicySurface:
    """Project a structured state without owning environment dynamics.

    This adapter is algorithm-neutral.  A hand-written policy, imitation
    learner, world model, PPO implementation, or native transfer evaluator can
    all use the same observation and legality projection.
    """

    runtime_config: Any
    spec: EnvironmentSpec

    @classmethod
    def from_contract(
        cls,
        structured_spec: EnvironmentSpec,
        runtime_config: Any,
        *,
        observation_size: int,
        action_size: int,
    ) -> "DensePolicySurface":
        if structured_spec.is_dense:
            raise ValueError("policy surface expects a structured environment spec")
        return cls(
            runtime_config=runtime_config,
            spec=EnvironmentSpec(
                observation_schema=ARSENAL_POLICY_SCHEMA,
                action_schema="hytalerl_reference_factored_arsenal_action_v2",
                action_components=structured_spec.action_components,
                observation_size=observation_size,
                action_size=action_size,
            ),
        )

    def project(self, state: Any, observation: Any) -> PolicyInput:
        """Return the exact dense observation and current factored mask."""

        action_surface = state.action_surface
        dense = arsenal_policy_observation(
            observation,
            action_surface.block_candidates,
            action_surface.recipe_encoding,
            inventory_tokens=state.inventory_tokens,
            light_tokens=state.light_tokens,
        )
        mask = state.action_mask
        if mask is None:
            raise RuntimeError("environment state did not cache its action mask")
        if dense.shape[-1] != self.spec.observation_size:
            raise RuntimeError("dense policy projection disagrees with its spec")
        if mask.shape[-1] != self.spec.action_size:
            raise RuntimeError("policy action mask disagrees with its spec")
        return PolicyInput(dense, mask)


__all__ = ["DensePolicySurface", "PolicyInput"]
