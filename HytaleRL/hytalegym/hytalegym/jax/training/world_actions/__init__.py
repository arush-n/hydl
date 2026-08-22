"""World-owned composition for learner-visible action candidates."""

from hytalegym.jax.training.world_actions.block_candidates import (
    NATIVE_ADVENTURE_DEFAULT_USE_DISTANCE,
    NATIVE_BLOCK_INTERACTION_DISTANCE_BUFFER,
    NATIVE_CAMERA_BLOCK_RAY_CELL_CAPACITY,
    NATIVE_CAMERA_BLOCK_RAY_DISTANCE,
    NATIVE_TYPED_TARGET_VIEW_SECTOR_RADIANS,
    NativeRegionBlockCandidateDefaults,
    make_native_region_runtime_block_candidate_provider,
    native_region_block_candidate_defaults,
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
