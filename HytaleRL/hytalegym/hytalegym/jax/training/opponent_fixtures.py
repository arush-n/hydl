"""Explicit opponent-start fixtures for training and evaluation."""

from __future__ import annotations

import jax.numpy as jnp

from hytalegym.jax.combat.types import CombatParams


def with_target_role_active_at_reset(params: CombatParams) -> CombatParams:
    """Start an existing target role immediately without changing its mechanics.

    The calibrated activation window describes a newly spawned native role
    graph. This overlay represents the distinct, native-reproducible task in
    which the target already exists and the player enters its encounter. It
    changes only the reset activation window; authored health, attacks,
    controller settings, movement, and timing remain untouched.
    """

    activation_tick = jnp.zeros_like(params.target_activation_min_tick)
    return params._replace(
        target_activation_min_tick=activation_tick,
        target_activation_max_tick=activation_tick,
    )


__all__ = ["with_target_role_active_at_reset"]
