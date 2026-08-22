"""Actor-safe Region action candidate producers."""

from hytalegym.jax.world.region.actions.camera_candidates import (
    NATIVE_CAMERA_BLOCK_RAY_CELL_CAPACITY,
    NATIVE_CAMERA_BLOCK_RAY_DISTANCE,
    REGION_CAMERA_BLOCK_CANDIDATE_SCHEMA,
    REGION_CAMERA_BLOCK_CANDIDATE_VERSION,
    produce_region_runtime_camera_block_action_candidates,
    region_camera_block_candidate_contract,
    region_camera_block_candidate_contract_sha256,
)

__all__ = [
    "NATIVE_CAMERA_BLOCK_RAY_CELL_CAPACITY",
    "NATIVE_CAMERA_BLOCK_RAY_DISTANCE",
    "REGION_CAMERA_BLOCK_CANDIDATE_SCHEMA",
    "REGION_CAMERA_BLOCK_CANDIDATE_VERSION",
    "produce_region_runtime_camera_block_action_candidates",
    "region_camera_block_candidate_contract",
    "region_camera_block_candidate_contract_sha256",
]
