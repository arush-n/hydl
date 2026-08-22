"""Host-side validation for fixed-shape observation inputs."""

from __future__ import annotations

import operator
from collections.abc import Iterable
from typing import Any

import numpy as np

from hytalegym.jax.combat.observation.v1.schema.contract import (
    DOOR_INTENT_COUNT,
    ENTITY_FLOAT_FEATURES,
    ENTITY_FLOAT_SIZE,
    ENTITY_INTEGER_SIZE,
    HAZARD_CAPACITY,
    HAZARD_FLOAT_FEATURES,
    HAZARD_FLOAT_SIZE,
    HAZARD_INTEGER_SIZE,
    INTERACTION_CAPACITY,
    INTERACTION_FLOAT_FEATURES,
    INTERACTION_FLOAT_SIZE,
    NEARBY_ENTITY_CAPACITY,
    PROJECTILE_CAPACITY,
    PROJECTILE_FLOAT_FEATURES,
    PROJECTILE_FLOAT_SIZE,
    PROJECTILE_INTEGER_SIZE,
    TERRAIN_FLOAT_FEATURES,
    TERRAIN_FLOAT_SIZE,
    TERRAIN_TOKEN_CAPACITY,
    TRAVERSAL_FLOAT_FEATURES,
    TRAVERSAL_FLOAT_SIZE,
    TRAVERSAL_TOKEN_CAPACITY,
)
from hytalegym.jax.combat.observation.v1.schema.types import (
    CombatSceneFeatures,
    InjectedWorldFeatures,
)


def validate_observation_inputs(
    scene: CombatSceneFeatures,
    world: InjectedWorldFeatures,
    *,
    batch_size: int,
) -> None:
    """Reject shape or dtype drift before values enter a compiled rollout."""

    if isinstance(batch_size, bool):
        raise TypeError("batch_size must be a positive integer")
    try:
        batch = operator.index(batch_size)
    except TypeError as error:
        raise TypeError("batch_size must be a positive integer") from error
    if batch <= 0:
        raise ValueError("batch_size must be a positive integer")
    _validate_arrays(
        (
            (
                "scene.entity_f32",
                scene.entity_f32,
                (batch, NEARBY_ENTITY_CAPACITY, ENTITY_FLOAT_SIZE),
                np.float32,
            ),
            (
                "scene.entity_i32",
                scene.entity_i32,
                (batch, NEARBY_ENTITY_CAPACITY, ENTITY_INTEGER_SIZE),
                np.int32,
            ),
            (
                "scene.entity_mask",
                scene.entity_mask,
                (batch, NEARBY_ENTITY_CAPACITY),
                np.bool_,
            ),
            (
                "scene.entity_overflow",
                scene.entity_overflow,
                (batch,),
                np.bool_,
            ),
            (
                "scene.projectile_f32",
                scene.projectile_f32,
                (batch, PROJECTILE_CAPACITY, PROJECTILE_FLOAT_SIZE),
                np.float32,
            ),
            (
                "scene.projectile_i32",
                scene.projectile_i32,
                (batch, PROJECTILE_CAPACITY, PROJECTILE_INTEGER_SIZE),
                np.int32,
            ),
            (
                "scene.projectile_mask",
                scene.projectile_mask,
                (batch, PROJECTILE_CAPACITY),
                np.bool_,
            ),
            (
                "scene.projectile_overflow",
                scene.projectile_overflow,
                (batch,),
                np.bool_,
            ),
            (
                "scene.hazard_f32",
                scene.hazard_f32,
                (batch, HAZARD_CAPACITY, HAZARD_FLOAT_SIZE),
                np.float32,
            ),
            (
                "scene.hazard_i32",
                scene.hazard_i32,
                (batch, HAZARD_CAPACITY, HAZARD_INTEGER_SIZE),
                np.int32,
            ),
            (
                "scene.hazard_mask",
                scene.hazard_mask,
                (batch, HAZARD_CAPACITY),
                np.bool_,
            ),
            (
                "scene.hazard_overflow",
                scene.hazard_overflow,
                (batch,),
                np.bool_,
            ),
        )
    )
    _validate_arrays(
        (
            (
                "world.terrain_f32",
                world.terrain_f32,
                (batch, TERRAIN_TOKEN_CAPACITY, TERRAIN_FLOAT_SIZE),
                np.float32,
            ),
            (
                "world.terrain_semantic_id",
                world.terrain_semantic_id,
                (batch, TERRAIN_TOKEN_CAPACITY),
                np.int32,
            ),
            (
                "world.terrain_flags",
                world.terrain_flags,
                (batch, TERRAIN_TOKEN_CAPACITY),
                np.uint32,
            ),
            (
                "world.terrain_mask",
                world.terrain_mask,
                (batch, TERRAIN_TOKEN_CAPACITY),
                np.bool_,
            ),
            (
                "world.terrain_overflow",
                world.terrain_overflow,
                (batch,),
                np.bool_,
            ),
            (
                "world.traversal_f32",
                world.traversal_f32,
                (batch, TRAVERSAL_TOKEN_CAPACITY, TRAVERSAL_FLOAT_SIZE),
                np.float32,
            ),
            (
                "world.traversal_id",
                world.traversal_id,
                (batch, TRAVERSAL_TOKEN_CAPACITY),
                np.int32,
            ),
            (
                "world.traversal_flags",
                world.traversal_flags,
                (batch, TRAVERSAL_TOKEN_CAPACITY),
                np.uint32,
            ),
            (
                "world.traversal_mask",
                world.traversal_mask,
                (batch, TRAVERSAL_TOKEN_CAPACITY),
                np.bool_,
            ),
            (
                "world.traversal_overflow",
                world.traversal_overflow,
                (batch,),
                np.bool_,
            ),
            (
                "world.interaction_f32",
                world.interaction_f32,
                (batch, INTERACTION_CAPACITY, INTERACTION_FLOAT_SIZE),
                np.float32,
            ),
            (
                "world.interaction_object_id",
                world.interaction_object_id,
                (batch, INTERACTION_CAPACITY),
                np.int32,
            ),
            (
                "world.interaction_is_door",
                world.interaction_is_door,
                (batch, INTERACTION_CAPACITY),
                np.bool_,
            ),
            (
                "world.interaction_door_intent_mask",
                world.interaction_door_intent_mask,
                (batch, INTERACTION_CAPACITY, DOOR_INTENT_COUNT),
                np.bool_,
            ),
            (
                "world.interaction_mask",
                world.interaction_mask,
                (batch, INTERACTION_CAPACITY),
                np.bool_,
            ),
            (
                "world.interaction_overflow",
                world.interaction_overflow,
                (batch,),
                np.bool_,
            ),
        )
    )
    _validate_semantics(scene, world)


def _validate_arrays(
    specifications: Iterable[tuple[str, Any, tuple[int, ...], Any]],
) -> None:
    for name, value, shape, dtype in specifications:
        actual_shape = tuple(value.shape)
        if actual_shape != shape:
            raise ValueError(f"{name} shape {actual_shape} does not match {shape}")
        actual_dtype = np.dtype(value.dtype)
        expected_dtype = np.dtype(dtype)
        if actual_dtype != expected_dtype:
            raise TypeError(
                f"{name} dtype {actual_dtype} does not match {expected_dtype}"
            )


def _validate_semantics(
    scene: CombatSceneFeatures,
    world: InjectedWorldFeatures,
) -> None:
    categories = (
        (
            "scene.entity",
            scene.entity_f32,
            scene.entity_mask,
            scene.entity_overflow,
            ENTITY_FLOAT_FEATURES,
        ),
        (
            "scene.projectile",
            scene.projectile_f32,
            scene.projectile_mask,
            scene.projectile_overflow,
            PROJECTILE_FLOAT_FEATURES,
        ),
        (
            "scene.hazard",
            scene.hazard_f32,
            scene.hazard_mask,
            scene.hazard_overflow,
            HAZARD_FLOAT_FEATURES,
        ),
        (
            "world.terrain",
            world.terrain_f32,
            world.terrain_mask,
            world.terrain_overflow,
            TERRAIN_FLOAT_FEATURES,
        ),
        (
            "world.traversal",
            world.traversal_f32,
            world.traversal_mask,
            world.traversal_overflow,
            TRAVERSAL_FLOAT_FEATURES,
        ),
        (
            "world.interaction",
            world.interaction_f32,
            world.interaction_mask,
            world.interaction_overflow,
            INTERACTION_FLOAT_FEATURES,
        ),
    )
    effective_masks: dict[str, np.ndarray] = {}
    for name, values, mask, overflow, feature_names in categories:
        effective = _effective_mask(mask, overflow)
        effective_masks[name] = effective
        _validate_compact_prefix(name, effective)
        _validate_active_f32(name, values, effective, feature_names)

    interaction_mask = effective_masks["world.interaction"]
    interaction_is_door = np.asarray(world.interaction_is_door)
    intent_mask = np.asarray(world.interaction_door_intent_mask)
    invalid_intent = (
        intent_mask & interaction_mask[:, :, None] & ~interaction_is_door[:, :, None]
    )
    if np.any(invalid_intent):
        batch, slot, intent = _first_index(invalid_intent)
        raise ValueError(
            "world.interaction_door_intent_mask is true for active "
            f"non-door candidate at batch {batch}, slot {slot}, "
            f"intent {intent}"
        )


def _effective_mask(mask: Any, overflow: Any) -> np.ndarray:
    mask_array = np.asarray(mask, dtype=np.bool_)
    overflow_array = np.asarray(overflow, dtype=np.bool_)
    return mask_array & ~overflow_array[:, None]


def _validate_compact_prefix(name: str, mask: np.ndarray) -> None:
    gap = ~mask[:, :-1] & mask[:, 1:]
    if np.any(gap):
        batch, prior_slot = _first_index(gap)
        raise ValueError(
            f"{name}_mask must be a compact prefix; batch {batch} has "
            f"an active slot after slot {prior_slot}"
        )


def _validate_active_f32(
    name: str,
    values: Any,
    mask: np.ndarray,
    feature_names: tuple[str, ...],
) -> None:
    array = np.asarray(values)
    active = mask[:, :, None]
    nonfinite = active & ~np.isfinite(array)
    if np.any(nonfinite):
        batch, slot, feature = _first_index(nonfinite)
        raise ValueError(
            f"{name}_f32 contains a non-finite active value at batch "
            f"{batch}, slot {slot}, feature {feature_names[feature]!r}"
        )
    out_of_range = active & ((array < -1.0) | (array > 1.0))
    if np.any(out_of_range):
        batch, slot, feature = _first_index(out_of_range)
        raise ValueError(
            f"{name}_f32 active value is outside [-1,1] at batch "
            f"{batch}, slot {slot}, feature {feature_names[feature]!r}: "
            f"{array[batch, slot, feature]}"
        )


def _first_index(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in np.argwhere(mask)[0])
