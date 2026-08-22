"""Compatibility facade for the motion navigation adapter."""

from hytalegym.jax.combat.motion.navigation import (
    TargetNavigationStep,
    validate_target_navigation_step,
)

__all__ = ["TargetNavigationStep", "validate_target_navigation_step"]
