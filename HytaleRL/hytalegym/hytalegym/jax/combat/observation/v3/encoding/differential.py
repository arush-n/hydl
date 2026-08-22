"""Host-side localization for dense Arsenal policy-row differentials."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from hytalegym.jax.combat.arsenal.schema.contract import OBSERVATION_CAPACITY
from hytalegym.jax.combat.mechanics import RESOURCE_COUNT, STATUS_CAPACITY
from hytalegym.jax.combat.observation.v1.schema.contract import (
    SELF_FLOAT_FEATURES,
    TARGET_FLOAT_FEATURES,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.blocks import (
    BLOCK_ACTION_CANDIDATE_POLICY_FEATURE_SIZE,
)
from hytalegym.jax.combat.observation.v3.schema.contract import (
    ABILITY_FLOAT_SIZE,
    ACTOR_WORLD_FLOAT_SIZE,
    ACTOR_WORLD_MASK_SIZE,
    DEFENSE_FLOAT_SIZE,
    DODGE_ACTION_COUNT,
    MOVEMENT_STATE_SIZE,
    STATUS_FLOAT_SIZE,
)
from hytalegym.jax.combat.observation.v3.tokens.inventory import (
    inventory_policy_flat_size,
)
from hytalegym.jax.combat.observation.v3.tokens.light import (
    actor_light_policy_flat_size,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    ARSENAL_POLICY_COMBAT_FLOAT_FEATURES,
    ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY,
    ARSENAL_POLICY_RECIPE_CANDIDATE_EMBEDDING_SIZE,
    arsenal_policy_observation_size,
)
from hytalegym.jax.combat.observation.v3.tokens.world import (
    WorldGeometryPolicyConfig,
    normalize_world_geometry_policy_config,
    world_geometry_policy_flat_size,
)
from hytalegym.jax.combat.skills import SKILL_COUNT
from hytalegym.jax.world import ACTOR_RECIPE_CANDIDATE_CAPACITY


def arsenal_policy_observation_group_slices(
    world_geometry_config: WorldGeometryPolicyConfig | None = None,
) -> tuple[tuple[str, slice], ...]:
    """Return named slices in the exact order used by the dense encoder."""

    token_config = normalize_world_geometry_policy_config(world_geometry_config)
    sizes = (
        ("self_f32", len(SELF_FLOAT_FEATURES)),
        ("target_f32", len(TARGET_FLOAT_FEATURES)),
        ("target_mask", 1),
        ("combat_f32", len(ARSENAL_POLICY_COMBAT_FLOAT_FEATURES)),
        ("resource_f32", RESOURCE_COUNT),
        ("resource_mask", RESOURCE_COUNT),
        ("defense_f32", DEFENSE_FLOAT_SIZE),
        ("status_f32", STATUS_CAPACITY * STATUS_FLOAT_SIZE),
        ("status_mask", STATUS_CAPACITY),
        ("ability_f32", OBSERVATION_CAPACITY * ABILITY_FLOAT_SIZE),
        ("ability_mask", OBSERVATION_CAPACITY),
        ("ability_legal", OBSERVATION_CAPACITY),
        ("actor_world_f32", ACTOR_WORLD_FLOAT_SIZE),
        ("actor_world_mask", ACTOR_WORLD_MASK_SIZE),
        ("movement_state_f32", MOVEMENT_STATE_SIZE),
        ("movement_state_mask", MOVEMENT_STATE_SIZE),
        ("world_geometry", world_geometry_policy_flat_size(token_config)),
        (
            "actor_light",
            actor_light_policy_flat_size(token_config.token_capacity),
        ),
        ("inventory", inventory_policy_flat_size()),
        ("skill_action_mask", SKILL_COUNT),
        ("jump_action_mask", 1),
        ("guard_action_mask", 1),
        ("dodge_action_mask", DODGE_ACTION_COUNT),
        ("observation_valid", 1),
        (
            "block_candidate_f32",
            ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY
            * BLOCK_ACTION_CANDIDATE_POLICY_FEATURE_SIZE,
        ),
        (
            "block_candidate_mask",
            ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY,
        ),
        ("block_candidates_available", 1),
        (
            "recipe_candidate_embedding",
            ACTOR_RECIPE_CANDIDATE_CAPACITY
            * ARSENAL_POLICY_RECIPE_CANDIDATE_EMBEDDING_SIZE,
        ),
        ("recipe_candidate_mask", ACTOR_RECIPE_CANDIDATE_CAPACITY),
        ("recipe_candidates_available", 1),
    )
    groups: list[tuple[str, slice]] = []
    offset = 0
    for name, size in sizes:
        groups.append((name, slice(offset, offset + size)))
        offset += size
    expected = arsenal_policy_observation_size(token_config)
    if offset != expected:
        raise RuntimeError(
            "policy observation group layout drift: "
            f"groups={offset} contract={expected}"
        )
    return tuple(groups)


def arsenal_policy_action_group_slices() -> tuple[tuple[str, slice], ...]:
    """Return named slices in the exact flattened action-mask order."""

    groups: list[tuple[str, slice]] = []
    offset = 0
    for name, size in zip(
        ARSENAL_POLICY_ACTION_HEAD_NAMES,
        ARSENAL_POLICY_ACTION_HEAD_SIZES,
        strict=True,
    ):
        groups.append((name, slice(offset, offset + size)))
        offset += size
    return tuple(groups)


def dense_row_differential(
    reference: Sequence[float] | np.ndarray,
    candidate: Sequence[float] | np.ndarray,
    *,
    groups: Sequence[tuple[str, slice]],
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
    absolute_tolerance_by_group: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Compare one dense row and localize every mismatch by named group."""

    left = np.asarray(reference)
    right = np.asarray(candidate)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("dense differential inputs must both be rank one")
    if left.shape != right.shape:
        raise ValueError(
            f"dense differential shapes differ: {left.shape} != {right.shape}"
        )
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("dense differential tolerances must be nonnegative")
    group_names = {name for name, _ in groups}
    if absolute_tolerance_by_group is None:
        absolute_tolerance_by_group = {}
    unknown_tolerances = set(absolute_tolerance_by_group) - group_names
    if unknown_tolerances:
        raise ValueError(
            "dense differential group tolerances name unknown groups: "
            + ", ".join(sorted(unknown_tolerances))
        )
    if any(value < 0.0 for value in absolute_tolerance_by_group.values()):
        raise ValueError("dense differential tolerances must be nonnegative")

    covered = np.zeros(left.shape, dtype=np.bool_)
    mismatches: list[dict[str, object]] = []
    total = 0
    maximum = 0.0
    for name, group_slice in groups:
        indexes = np.arange(left.size)[group_slice]
        if indexes.size == 0:
            raise ValueError(f"dense differential group {name!r} is empty")
        if np.any(covered[indexes]):
            raise ValueError(f"dense differential group {name!r} overlaps")
        covered[indexes] = True
        equal = np.isclose(
            left[indexes],
            right[indexes],
            rtol=relative_tolerance,
            atol=absolute_tolerance_by_group.get(name, absolute_tolerance),
            equal_nan=False,
        )
        local = np.flatnonzero(~equal)
        if local.size == 0:
            continue
        absolute = indexes[local]
        # NumPy deliberately rejects boolean subtraction.  Action masks were
        # previously safe only while they happened to be equal, so the first
        # real mask divergence crashed the diagnostic instead of localizing
        # it.  Promote numeric rows before subtraction and represent a
        # boolean disagreement as unit error.
        if np.issubdtype(left.dtype, np.bool_) and np.issubdtype(
            right.dtype,
            np.bool_,
        ):
            absolute_error = np.not_equal(
                left[absolute],
                right[absolute],
            ).astype(np.float64)
        else:
            absolute_error = np.abs(
                left[absolute].astype(np.float64)
                - right[absolute].astype(np.float64)
            )
        finite_error = absolute_error[np.isfinite(absolute_error)]
        group_maximum = (
            float(np.max(finite_error)) if finite_error.size else float("inf")
        )
        maximum = max(maximum, group_maximum)
        total += int(local.size)
        mismatches.append(
            {
                "group": name,
                "count": int(local.size),
                "maximum_absolute_error": group_maximum,
                "absolute_indices": absolute[:16].tolist(),
                "reference_values": left[absolute[:16]].tolist(),
                "candidate_values": right[absolute[:16]].tolist(),
            }
        )
    if not bool(np.all(covered)):
        missing = np.flatnonzero(~covered)
        raise ValueError(
            "dense differential groups do not cover the complete row: "
            f"first_missing={int(missing[0])}"
        )
    return {
        "width": int(left.size),
        "mismatch_count": total,
        "maximum_absolute_error": maximum,
        "groups": mismatches,
    }


__all__ = [
    "arsenal_policy_action_group_slices",
    "arsenal_policy_observation_group_slices",
    "dense_row_differential",
]
