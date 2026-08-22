"""Native-sourced defaults for exact Region block-action candidates."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from hytalegym.jax.combat.arsenal.environment import (
    ArsenalBlockCandidateProvider,
    make_region_runtime_camera_block_candidate_provider,
)
from hytalegym.jax.world.region.actions import (
    NATIVE_CAMERA_BLOCK_RAY_CELL_CAPACITY,
    NATIVE_CAMERA_BLOCK_RAY_DISTANCE,
)


# InteractionConfiguration.DEFAULT authors Adventure UseDistance=5.0.  Both
# InteractionValidation and the bridge's NativeBlockUse add 2.0 before their
# exact distance check.  Keep the two terms separate so an item-authored
# UseDistance override cannot accidentally receive the already-buffered 7.0.
NATIVE_ADVENTURE_DEFAULT_USE_DISTANCE = 5.0
NATIVE_BLOCK_INTERACTION_DISTANCE_BUFFER = 2.0

# Retained for callers that deliberately request the exhaustive dense
# mechanics surface. It is not the native player-camera target selector.
NATIVE_TYPED_TARGET_VIEW_SECTOR_RADIANS = math.tau


class NativeRegionBlockCandidateDefaults(NamedTuple):
    """Derived native camera and commit-time Region targeting limits."""

    maximum_distance: float
    view_sector_full_angle_radians: float
    cell_radius: int
    ray_distance: float
    ray_cell_capacity: int


def native_region_block_candidate_defaults(
    authored_adventure_use_distance: float = (
        NATIVE_ADVENTURE_DEFAULT_USE_DISTANCE
    ),
) -> NativeRegionBlockCandidateDefaults:
    """Derive native player-camera targeting and commit-time reach.

    ``authored_adventure_use_distance`` is the held item's resolved
    ``InteractionConfiguration.UseDistance[Adventure]``.  Omitting it is
    correct only for the native default configuration.  Item overrides must
    be supplied from resolved asset or native evidence.
    """

    if (
        isinstance(authored_adventure_use_distance, bool)
        or not isinstance(
            authored_adventure_use_distance,
            (int, float, np.integer, np.floating),
        )
        or not math.isfinite(float(authored_adventure_use_distance))
        or float(authored_adventure_use_distance) <= 0.0
    ):
        raise ValueError(
            "authored_adventure_use_distance must be finite and positive"
        )
    maximum = (
        float(authored_adventure_use_distance)
        + NATIVE_BLOCK_INTERACTION_DISTANCE_BUFFER
    )
    return NativeRegionBlockCandidateDefaults(
        maximum_distance=maximum,
        view_sector_full_angle_radians=(
            NATIVE_TYPED_TARGET_VIEW_SECTOR_RADIANS
        ),
        # A dense integer-cell stencil is a coverage boundary, so fractional
        # reach always rounds outward rather than truncating. These two dense
        # fields remain for explicit mechanics/debug callers; the native
        # production factory below uses the camera fields.
        cell_radius=math.ceil(maximum),
        ray_distance=NATIVE_CAMERA_BLOCK_RAY_DISTANCE,
        ray_cell_capacity=NATIVE_CAMERA_BLOCK_RAY_CELL_CAPACITY,
    )


def make_native_region_runtime_block_candidate_provider(
    *,
    authored_adventure_use_distance: float = (
        NATIVE_ADVENTURE_DEFAULT_USE_DISTANCE
    ),
) -> ArsenalBlockCandidateProvider:
    """Build the native camera-target Region provider from authored reach.

    The producer walks the native eight-block eye/head presence ray, selects
    its first current non-air block, then applies the separately authored
    interaction-distance check. Native camera targeting does not consult the
    NPC perception role-opacity predicate.
    """

    defaults = native_region_block_candidate_defaults(
        authored_adventure_use_distance
    )
    return make_region_runtime_camera_block_candidate_provider(
        maximum_interaction_distance=defaults.maximum_distance,
        ray_distance=defaults.ray_distance,
        ray_cell_capacity=defaults.ray_cell_capacity,
    )


__all__ = [
    "NATIVE_ADVENTURE_DEFAULT_USE_DISTANCE",
    "NATIVE_BLOCK_INTERACTION_DISTANCE_BUFFER",
    "NATIVE_CAMERA_BLOCK_RAY_CELL_CAPACITY",
    "NATIVE_CAMERA_BLOCK_RAY_DISTANCE",
    "NATIVE_TYPED_TARGET_VIEW_SECTOR_RADIANS",
    "NativeRegionBlockCandidateDefaults",
    "make_native_region_runtime_block_candidate_provider",
    "native_region_block_candidate_defaults",
]
