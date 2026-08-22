"""Host-side shape and semantic validation for effect commands."""

from __future__ import annotations

from typing import Any

import numpy as np

from hytalegym.jax.combat.effects.schema.types import CombatEffectCommands


def validate_effect_commands(
    commands: CombatEffectCommands,
    *,
    batch_size: int | None = None,
) -> None:
    """Raise before ``device_put`` when an authored command is malformed."""

    requested = np.asarray(commands.projectile.requested)
    batch = requested.shape[0] if requested.ndim == 1 else -1
    if batch_size is not None and batch != batch_size:
        raise ValueError(f"effect command batch {batch} does not match {batch_size}")
    _validate_tree_shapes(commands, batch)

    projectile = commands.projectile
    p_mask = np.asarray(projectile.requested, dtype=np.bool_)
    _finite(p_mask, "projectile.position", projectile.position)
    _finite(p_mask, "projectile.velocity", projectile.velocity)
    _bounded(
        p_mask[:, None],
        "projectile.half_extent",
        projectile.half_extent,
        minimum=np.nextafter(np.float32(0.0), np.float32(1.0)),
    )
    for name in (
        "despawn_seconds",
        "authored_lifetime_seconds",
        "terminal_velocity",
        "velocity_scale",
    ):
        _bounded(
            p_mask,
            f"projectile.{name}",
            getattr(projectile, name),
            minimum=np.nextafter(np.float32(0.0), np.float32(1.0)),
        )
    for name in ("damage", "gravity", "dead_time_seconds"):
        _bounded(
            p_mask,
            f"projectile.{name}",
            getattr(projectile, name),
            minimum=0.0,
        )
    if np.any(p_mask & (np.asarray(projectile.kind) <= 0)):
        raise ValueError("requested projectile.kind must be positive")

    hazard = commands.hazard
    h_mask = np.asarray(hazard.requested, dtype=np.bool_)
    _finite(h_mask, "hazard.center", hazard.center)
    _bounded(
        h_mask[:, None],
        "hazard.half_extent",
        hazard.half_extent,
        minimum=0.0,
    )
    _bounded(
        h_mask,
        "hazard.duration_seconds",
        hazard.duration_seconds,
        minimum=np.nextafter(np.float32(0.0), np.float32(1.0)),
    )
    _bounded(
        h_mask,
        "hazard.damage_per_second",
        hazard.damage_per_second,
        minimum=0.0,
    )
    for name in ("intensity", "activation"):
        _bounded(
            h_mask,
            f"hazard.{name}",
            getattr(hazard, name),
            minimum=0.0,
            maximum=1.0,
        )
    if np.any(h_mask & (np.asarray(hazard.kind) <= 0)):
        raise ValueError("requested hazard.kind must be positive")


def _validate_tree_shapes(commands: CombatEffectCommands, batch: int) -> None:
    if batch <= 0:
        raise ValueError("effect commands require a positive batch")
    vectors = {
        "projectile.position": commands.projectile.position,
        "projectile.velocity": commands.projectile.velocity,
        "projectile.half_extent": commands.projectile.half_extent,
        "hazard.center": commands.hazard.center,
        "hazard.half_extent": commands.hazard.half_extent,
    }
    for name, value in vectors.items():
        if np.shape(value) != (batch, 3):
            raise ValueError(f"{name} must have shape ({batch}, 3)")
        _require_dtype(name, value, np.float32)
    for group_name, group in (
        ("projectile", commands.projectile),
        ("hazard", commands.hazard),
    ):
        for field, value in zip(group._fields, group, strict=True):
            if field in {"position", "velocity", "half_extent", "center"}:
                continue
            if np.shape(value) != (batch,):
                raise ValueError(f"{group_name}.{field} must have shape ({batch},)")
    for name, value in (
        ("projectile.requested", commands.projectile.requested),
        ("projectile.hostile", commands.projectile.hostile),
        (
            "projectile.entity_collision_only",
            commands.projectile.entity_collision_only,
        ),
        ("hazard.requested", commands.hazard.requested),
        ("hazard.hostile", commands.hazard.hostile),
        (
            "hazard.entity_overlap_only",
            commands.hazard.entity_overlap_only,
        ),
    ):
        _require_dtype(name, value, np.bool_)
    for group_name, group in (
        ("projectile", commands.projectile),
        ("hazard", commands.hazard),
    ):
        for field, value in zip(group._fields, group, strict=True):
            name = f"{group_name}.{field}"
            if field in {
                "requested",
                "hostile",
                "entity_collision_only",
                "entity_overlap_only",
                "position",
                "velocity",
                "half_extent",
                "center",
            }:
                continue
            expected = (
                np.uint32
                if field == "flags"
                else np.int32
                if field in {"kind", "owner_entity_id"}
                else np.float32
            )
            _require_dtype(name, value, expected)


def _finite(mask: np.ndarray, name: str, value: Any) -> None:
    array = np.asarray(value)
    expanded = mask.reshape(mask.shape + (1,) * (array.ndim - mask.ndim))
    if np.any(expanded & ~np.isfinite(array)):
        raise ValueError(f"{name} must be finite for requested rows")


def _bounded(
    mask: np.ndarray,
    name: str,
    value: Any,
    *,
    minimum: float,
    maximum: float | None = None,
) -> None:
    array = np.asarray(value)
    expanded = np.broadcast_to(mask, array.shape)
    invalid = ~np.isfinite(array) | (array < minimum)
    if maximum is not None:
        invalid |= array > maximum
    if np.any(expanded & invalid):
        raise ValueError(f"{name} is outside its supported range")


def _require_dtype(name: str, value: Any, expected: Any) -> None:
    actual = np.asarray(value).dtype
    if actual != np.dtype(expected):
        raise TypeError(f"{name} must have dtype {np.dtype(expected)}, got {actual}")
