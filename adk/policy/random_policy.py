"""Uniform-legal policy.

Not a strategy -- it is the ADK's driver for plumbing tests and the control
arm every evaluation must report against. Uniform-legal has come within 1.73
points of a trained policy, so it is a real bar, not a floor.
"""

from __future__ import annotations

import jax.numpy as jnp

from adk.runtime.env_adapter import legal_mask, sample_actions


def uniform_legal(carry, observation, key):
    """Sample uniformly over legal actions. Stateless: carry passes through."""

    mask = legal_mask(observation)
    logits = jnp.zeros(mask.shape, dtype=jnp.float32)
    return carry, sample_actions(logits, mask, key)
