"""Stable policy seam exposed to agent authors.

A policy owns strategy and recurrent state. The ADK supplies the exact
structured legal observation alongside its unchanged dense bridge projection,
then owns mask/factor transport, environment stepping, and lifecycle around it.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax

from adk.architecture.inputs import ActorPolicyInput
from adk.runtime.env_adapter import PolicyInput


@runtime_checkable
class Policy(Protocol):
    """Callable accepted by rollout and evaluation tools.

    The returned action is always explicit ``int32[B, H]`` factors.  Policies
    must use ``policy_input.action_mask``; the environment and native session
    reject masked factors rather than converting them to a penalty.
    ``policy_input.legal_observation`` is the upstream structured learner
    PyTree and contains no privileged simulator state.
    """

    def __call__(
        self,
        carry: Any,
        policy_input: ActorPolicyInput,
        key: jax.Array,
        /,
    ) -> tuple[Any, jax.Array]: ...


__all__ = [
    "ActorPolicyInput",
    "Policy",
    "PolicyInput",
]
