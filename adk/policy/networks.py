"""Named, swappable policy networks.

A network was previously implicit: an agent that wanted something other than the
recurrent PPO network had to hand-build both the parameters and the policy
closure, and there was no way to name the pairing or swap it from a spec.

A :class:`Network` is exactly that pairing -- how to initialise parameters, and
how to turn parameters into a :class:`Policy`.  Nothing here knows about a
training algorithm, so the same registry serves gradient descent, evolutionary
search, and fixed controllers.

Two networks are registered because a registry with no entries is a shape
without an implementation:

``uniform_legal``
    Parameterless. The ADK's existing control arm, exposed under the same seam.
``linear``
    One dense layer from the observation to the action logits, masked. The
    smallest thing that actually has parameters to train.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from adk.policy import Policy
from adk.policy.random_policy import uniform_legal
from adk.runtime.env_adapter import legal_mask, sample_actions


class Network(NamedTuple):
    """How to create parameters, and how to act with them."""

    initialize: Callable[..., Any]
    policy_of: Callable[[Any], Policy]


_NETWORKS: dict[str, Network] = {}


def register_network(name: str, network: Network) -> Network:
    """Register one named network, refusing to silently replace another."""

    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("network name must be a non-empty bare string")
    if not isinstance(network, Network):
        raise TypeError("network must be a Network(initialize, policy_of)")
    existing = _NETWORKS.get(name)
    if existing is not None and existing != network:
        raise ValueError(f"network already registered under a different implementation: {name!r}")
    _NETWORKS[name] = network
    return network


def get_network(name: str) -> Network:
    if name not in _NETWORKS:
        raise KeyError(
            f"unknown network: {name!r}; registered: {sorted(_NETWORKS)}"
        )
    return _NETWORKS[name]


def list_networks() -> list[str]:
    return sorted(_NETWORKS)


def _initialize_uniform_legal(handle: Any, key: jax.Array) -> dict[str, Any]:
    """No parameters; the signature matches so networks stay swappable."""

    del handle, key
    return {}


def _uniform_legal_policy_of(parameters: Any) -> Policy:
    del parameters
    return uniform_legal


def _initialize_linear(handle: Any, key: jax.Array) -> dict[str, jax.Array]:
    width = handle.observation_size
    logits = sum(handle.action_head_sizes)
    scale = 1.0 / jnp.sqrt(jnp.asarray(width, dtype=jnp.float32))
    return {
        "w": jax.random.normal(key, (width, logits), dtype=jnp.float32) * scale,
        "b": jnp.zeros((logits,), dtype=jnp.float32),
    }


def _linear_policy_of(parameters: Any) -> Policy:
    def policy(carry, actor_input, key):
        mask = legal_mask(actor_input)
        logits = actor_input.observation @ parameters["w"] + parameters["b"]
        return carry, sample_actions(logits, mask, key)

    return policy


register_network(
    "uniform_legal",
    Network(_initialize_uniform_legal, _uniform_legal_policy_of),
)
register_network(
    "linear",
    Network(_initialize_linear, _linear_policy_of),
)
