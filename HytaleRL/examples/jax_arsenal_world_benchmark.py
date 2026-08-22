"""Benchmark exact Arsenal world queries against the explicit all-clear control."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.compilation_cache import (
    configure_persistent_compilation_cache,
)
from hytalegym.geometry.contract import (
    CELL_COUNT,
    FLAG_OPAQUE,
    FLUID_MOVEMENT_FEATURES,
    MOVEMENT_FEATURES,
)
from hytalegym.jax.combat import (
    ARSENAL_POLICY_OBSERVATION_SIZE,
    DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
    PROFILE_NAMES,
    arsenal_policy_contract_sha256,
    arsenal_policy_observation,
    arsenal_runtime_config,
    default_combat_params,
    empty_injected_world_features,
    encode_world_geometry_policy_tokens,
    hytale_0_5_7_loadouts,
    neutral_arsenal_policy_action_factors,
)
from hytalegym.jax.combat.arsenal.world_actions.executor import (
    compile_region_block_executor_assets,
    make_region_block_action_surface_executor,
    make_region_block_item_availability_provider,
    make_region_held_item_inventory_reset_provider,
)
from hytalegym.jax.combat.arsenal.world_actions.crafting import (
    compile_region_fieldcraft_assets,
    make_region_fieldcraft_candidate_provider,
    region_fieldcraft_contract_sha256,
)
from hytalegym.jax.combat.arsenal.environment import (
    REFERENCE_SKILL_MAXIMUM_TURN_DEGREES,
    item_interaction_evidence_from_native_recording,
    make_arsenal_environment,
)
from hytalegym.jax.combat.arsenal.training_inventory import (
    training_inventory_from_config,
)
from hytalegym.jax.combat.observation.v3.tokens.light import (
    encode_actor_light_policy_tokens,
)
from hytalegym.jax.training import (
    ArsenalResetBatch,
    ArsenalWorldRuntimeViews,
    geometry_arsenal_world_capabilities,
    geometry_arsenal_world_tokens,
    item_interaction_table_from_native_evidence,
    load_region_combat_reset_pool,
    make_arsenal_action_surface_provider,
    make_inventory_item_interaction_availability_provider,
    make_region_runtime_camera_block_candidate_provider,
    make_region_runtime_block_candidate_provider,
    make_region_combat_reset_provider,
    open_flat_arsenal_world_capabilities,
    region_arsenal_entity_only_explosion_candidates,
    region_reset_actor_evidence_contract,
    region_combat_reset_pool_sha256,
    region_combat_reset_core_admission_contract,
    region_combat_reset_core_admission_contract_sha256,
    region_reset_filter_combat_contract,
)
from hytalegym.jax.world import (
    COLLISION_SHAPE_NONE,
    REGION_ARTIFACT_USAGE_TRAINING,
    REGION_TRAVERSAL_POLICY_STAGING_CAPACITY,
    GeometryState,
    RegionActionRuntimeState,
    WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY,
    WORLD_TOKEN_PROVENANCE_SURROGATE_EXACT_GEOMETRY,
    actor_block_light_fallback_contract_sha256,
    native_or_surrogate_actor_block_light_tokens,
    produce_actor_region_world_geometry_tokens,
    geometry_perception_line_of_sight_result,
    entity_only_explosion_contract_sha256,
    region_action_runtime_contract_sha256,
    region_action_runtime_from_artifact_libraries,
    region_environments_from_artifact_manifest,
    REGION_COMBAT_SWEEP_CELL_CAPACITY,
    region_geometry_from_atlas,
    region_native_actor_block_light_tokens,
    region_native_light_atlas_contract_sha256,
    region_native_light_atlas_from_coverage_library,
    region_surrogate_actor_block_light_tokens,
    region_surrogate_sky_light_atlas_from_region,
    region_surrogate_sky_light_contract_sha256,
    world_geometry_token_contract_sha256,
)
from hytalegym.jax.world.region.actions import (
    NATIVE_CAMERA_BLOCK_RAY_CELL_CAPACITY,
    NATIVE_CAMERA_BLOCK_RAY_DISTANCE,
    region_camera_block_candidate_contract_sha256,
)
from hytalegym.jax.world.region.los import (
    REGION_PERCEPTION_LOS_DISTANCE_BLOCKS,
)
from hytalegym.jax.world.navigation.adapter import (
    surrogate_region_walk_navigation_contract_sha256,
    surrogate_region_walk_target_navigation_provider,
)
from hytalegym.worldgen.region import (
    CHUNK_SIZE,
    CORE_BLOCKS_PER_AXIS,
    current_native_evidence_jar_sha256,
)


_BENCHMARK_SCHEMA = "hytalerl_arsenal_world_policy_benchmark_v7"
_RESET_POOL_SEMANTIC_FIELDS = (
    "schema",
    "usage",
    "combat_filter_contract",
    "actor_evidence_contract",
    "region_library_semantic_sha256",
    "traversal_library_semantic_sha256",
    "working_set_capacity",
    "selection_key",
    "assignment_key",
    "edge_selection_seed",
    "edge_selection_method",
    "filter_seeds",
    "filter_steps",
    "filter_semantics",
    "task_bias",
    "worlds",
)


@dataclass(frozen=True)
class ArsenalRegionFixture:
    """Hash-pinned Regions bound to every Arsenal world/reset seam.

    World-action bindings are optional and independently fail closed. A
    fixture may publish them only when it owns a persistent mutable World
    runtime that can recheck selected targets and acknowledge committed
    changes.
    """

    params: Any
    geometry_provider: Any
    target_navigation_provider: Callable
    target_navigation_contract_sha256: str
    world_capability_provider: Callable
    world_token_provider: Callable
    metadata: dict[str, object]
    reset_provider: Callable
    inventory_reset_provider: Callable | None = None
    action_surface_provider: Callable | None = None
    action_surface_executor: Callable | None = None
    action_surface_runtime_initializer: Callable | None = None
    world_runtime_provider: Callable | None = None
    actor_world_runtime_provider: Callable | None = None
    explosion_candidate_provider: Callable | None = None


@dataclass(frozen=True)
class RegionActionSurfaceConfig:
    """Host-side inputs required to publish exact Region action evidence."""

    #: ``None`` when the bound Region library ships no block-semantic sidecars.
    #: The action-surface runtime is still built -- the rollout needs its
    #: mutable-physics half -- but every semantic mask is false, so the block
    #: interaction heads close instead of reading identities nothing captured.
    #: See ``region_block_semantic_atlas_without_evidence``.
    block_semantic_manifest: Path | None
    mutation_capacity: int
    maximum_distance: float
    view_sector_full_angle_degrees: float
    candidate_selection: str = "native_camera"
    native_item_interaction_recording: Path | None = None
    native_crafting_catalog_recording: Path | None = None
    assets_path: Path | None = None
    modification_allowed: bool = False
    held_item_asset_id: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.mutation_capacity, bool)
            or not isinstance(self.mutation_capacity, int)
            or self.mutation_capacity < 1
        ):
            raise ValueError("mutation_capacity must be a positive integer")
        for value, name in (
            (self.maximum_distance, "maximum_distance"),
            (
                self.view_sector_full_angle_degrees,
                "view_sector_full_angle_degrees",
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if float(self.view_sector_full_angle_degrees) > 360.0:
            raise ValueError(
                "view_sector_full_angle_degrees cannot exceed 360"
            )
        if self.candidate_selection not in {"native_camera", "dense_complete"}:
            raise ValueError(
                "candidate_selection must be native_camera or dense_complete"
            )
        if not isinstance(self.modification_allowed, bool):
            raise TypeError("modification_allowed must be boolean")
        if self.held_item_asset_id is not None and (
            not isinstance(self.held_item_asset_id, str)
            or not self.held_item_asset_id
        ):
            raise ValueError("held_item_asset_id must be a non-empty string")
        if self.held_item_asset_id is not None and not self.modification_allowed:
            raise ValueError(
                "held_item_asset_id requires modification_allowed"
            )
        if self.native_crafting_catalog_recording is not None and (
            self.native_item_interaction_recording is None
            or not self.modification_allowed
        ):
            raise ValueError(
                "fieldcraft execution currently requires the unified mutable "
                "Region executor"
            )


@dataclass(frozen=True)
class RegionNativeLightConfig:
    """Pinned native static-light coverage for an exact Region fixture."""

    coverage_manifest: Path
    expected_coverage_semantic_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.coverage_manifest, Path):
            raise TypeError("coverage_manifest must be a pathlib.Path")
        if not self.coverage_manifest.is_file():
            raise ValueError("coverage_manifest must name an existing file")
        value = self.expected_coverage_semantic_sha256
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                "expected_coverage_semantic_sha256 must be lowercase 64-hex"
            )


def _fixed_region_reset_provider(
    *,
    agent_position: jax.Array,
    target_position: jax.Array,
    batch_size: int,
) -> Callable[[jax.Array], ArsenalResetBatch]:
    """Bind one selected Region edge to every lane without relying on params."""

    agent = jnp.broadcast_to(
        jnp.asarray(agent_position, dtype=jnp.float32),
        (batch_size, 3),
    )
    target = jnp.broadcast_to(
        jnp.asarray(target_position, dtype=jnp.float32),
        (batch_size, 3),
    )

    def provider(keys: jax.Array) -> ArsenalResetBatch:
        if keys.ndim < 1 or keys.shape[0] != batch_size:
            raise ValueError(
                "fixed Region reset keys must preserve the fixture batch axis"
            )
        return ArsenalResetBatch(
            agent_position=agent,
            target_position=target,
        )

    return provider


def _region_action_surface_metadata(
    config: RegionActionSurfaceConfig | None,
) -> dict[str, object]:
    if config is None:
        return {
            "runtime_bound": False,
            "candidate_producer_bound": False,
            "item_root_evidence_bound": False,
            "executor_bound": False,
        }
    return {
        "runtime_bound": True,
        "runtime_contract_sha256": region_action_runtime_contract_sha256(),
        # None means the library ships no block-semantic sidecars, so the block
        # heads are closed. Reported as an explicit null rather than omitted:
        # a missing key reads as "not recorded", this reads as "recorded, and
        # there is nothing".
        "block_semantic_manifest_file_sha256": (
            None
            if config.block_semantic_manifest is None
            else _file_sha256(config.block_semantic_manifest)
        ),
        "block_semantics_bound": config.block_semantic_manifest is not None,
        "mutation_capacity": config.mutation_capacity,
        "maximum_distance": float(config.maximum_distance),
        "candidate_selection": config.candidate_selection,
        "camera_candidate_contract_sha256": (
            region_camera_block_candidate_contract_sha256()
            if config.candidate_selection == "native_camera"
            else None
        ),
        "camera_ray_distance": (
            NATIVE_CAMERA_BLOCK_RAY_DISTANCE
            if config.candidate_selection == "native_camera"
            else None
        ),
        "camera_ray_cell_capacity": (
            NATIVE_CAMERA_BLOCK_RAY_CELL_CAPACITY
            if config.candidate_selection == "native_camera"
            else None
        ),
        "view_sector_full_angle_degrees": (
            float(config.view_sector_full_angle_degrees)
            if config.candidate_selection == "dense_complete"
            else None
        ),
        "candidate_cell_radius": (
            math.ceil(config.maximum_distance)
            if config.candidate_selection == "dense_complete"
            else None
        ),
        "candidate_producer_bound": True,
        "item_root_evidence_bound": (
            config.native_item_interaction_recording is not None
        ),
        "item_root_recording_file_sha256": (
            None
            if config.native_item_interaction_recording is None
            else _file_sha256(config.native_item_interaction_recording)
        ),
        "fieldcraft_recording_file_sha256": (
            None
            if config.native_crafting_catalog_recording is None
            else _file_sha256(config.native_crafting_catalog_recording)
        ),
        "fieldcraft_contract_sha256": (
            None
            if config.native_crafting_catalog_recording is None
            else region_fieldcraft_contract_sha256()
        ),
        "unified_physics_projection_bound": True,
        "unified_projection_blocker": None,
        "executor_bound": bool(
            config.native_item_interaction_recording is not None
            and config.modification_allowed
        ),
        "scenario_held_item_asset_id": config.held_item_asset_id,
    }


def _region_explosion_metadata(
    config: RegionActionSurfaceConfig | None,
) -> dict[str, object]:
    bound = config is not None
    return {
        "provider_bound": bound,
        "contract_sha256": (
            entity_only_explosion_contract_sha256() if bound else None
        ),
        "scope": (
            "entity_only_current_region_physical_admission"
            if bound
            else "unbound"
        ),
    }


def _region_light_metadata(
    config: RegionNativeLightConfig | None,
) -> dict[str, object]:
    surrogate = {
        "provider_bound": True,
        "contract_sha256": region_surrogate_sky_light_contract_sha256(),
        "provenance": "surrogate_not_native_light_calibrated",
        "block_light_rgb_available": False,
    }
    if config is None:
        return {
            "mode": "surrogate_only",
            "native": None,
            "surrogate_fallback": surrogate,
        }
    coverage = json.loads(
        config.coverage_manifest.read_text(encoding="utf-8")
    )
    return {
        "mode": "native_static_with_complete_surrogate_row_fallback",
        "native": {
            "provider_bound": True,
            "atlas_contract_sha256": (
                region_native_light_atlas_contract_sha256()
            ),
            "coverage_manifest_file_sha256": _file_sha256(
                config.coverage_manifest
            ),
            "coverage_library_semantic_sha256": (
                config.expected_coverage_semantic_sha256
            ),
            "coverage_artifact_count": coverage.get("artifact_count"),
            "coverage_available_count": coverage.get("available_count"),
            "provenance": "native_static_capture",
            "channels": ["sky_light", "block_light_rgb"],
        },
        "fallback_contract_sha256": (
            actor_block_light_fallback_contract_sha256()
        ),
        "surrogate_fallback": surrogate,
    }


def load_arsenal_region_fixture(
    *,
    region_manifest: Path,
    traversal_manifest: Path,
    expected_region_sha256: str,
    expected_traversal_sha256: str,
    batch_size: int,
    selection_key: int,
    assignment_key: int,
    node_seed: int,
    params,
    runtime,
    target_separation_range: tuple[float, float] | None = None,
    require_initial_line_of_sight: bool = True,
    working_set_capacity: int = 1,
    environment_diversity: bool = False,
    selection_seeds: Sequence[int] | None = None,
    action_surface_config: RegionActionSurfaceConfig | None = None,
    native_light_config: RegionNativeLightConfig | None = None,
) -> ArsenalRegionFixture:
    """Load one exact training Region without regenerating a seed.

    ``working_set_capacity`` publishes that many library worlds and spreads the
    batch across them. ``environment_diversity`` additionally gives each row
    its own spawn/target pair inside its own world; without it every row shares
    one pair and the batch is byte-identical, which lets a policy memorise a
    single spawn instead of learning the task.

    ``selection_seeds`` names the worlds outright and overrides ``selection_key``.
    The key ranks the whole split by digest and takes a prefix, so it can only
    ever express "the N that hash first"; a deliberately composed set -- half
    flat terrain for learning the task, half varied for generalising -- is not
    reachable through it at any key. Slot order is preserved because
    ``environment_world_id`` indexes it.
    """

    separation_range = _target_separation_range(target_separation_range)

    arguments = argparse.Namespace(
        region_manifest=region_manifest,
        traversal_manifest=traversal_manifest,
        expected_region_sha256=expected_region_sha256,
        expected_traversal_sha256=expected_traversal_sha256,
        batch_size=batch_size,
        region_selection_key=selection_key,
        region_assignment_key=assignment_key,
        region_selection_seeds=(
            None if selection_seeds is None else tuple(int(s) for s in selection_seeds)
        ),
        region_node_seed=node_seed,
        region_target_separation_range=separation_range,
        region_require_initial_line_of_sight=bool(require_initial_line_of_sight),
        region_working_set_capacity=int(working_set_capacity),
        region_environment_diversity=bool(environment_diversity),
        region_action_surface_config=action_surface_config,
        region_native_light_config=native_light_config,
    )
    (
        region_params,
        geometry_provider,
        world_capability_provider,
        world_token_provider,
        metadata,
        action_surface_runtime_initializer,
        world_runtime_provider,
        actor_world_runtime_provider,
        action_surface_provider,
        action_surface_executor,
        inventory_reset_provider,
    ) = _region_case_with_geometry(arguments, params, runtime)
    # ``_fixed_region_reset_provider`` broadcasts to ``(batch, 3)``, so a
    # per-row ``(batch, 3)`` array passes through unchanged while the scalar
    # single-pair case still fans out to every lane.
    agent_position = metadata.get("environment_agent_position")
    target_position = metadata.get("environment_target_position")
    reset_provider = _fixed_region_reset_provider(
        agent_position=(
            region_params.agent_spawn
            if agent_position is None
            else jnp.asarray(agent_position, dtype=jnp.float32)
        ),
        target_position=(
            (region_params.agent_spawn + region_params.target_offset)
            if target_position is None
            else jnp.asarray(target_position, dtype=jnp.float32)
        ),
        batch_size=batch_size,
    )
    return ArsenalRegionFixture(
        params=region_params,
        geometry_provider=geometry_provider,
        target_navigation_provider=(
            surrogate_region_walk_target_navigation_provider
        ),
        target_navigation_contract_sha256=(
            surrogate_region_walk_navigation_contract_sha256()
        ),
        world_capability_provider=world_capability_provider,
        world_token_provider=world_token_provider,
        reset_provider=reset_provider,
        action_surface_runtime_initializer=(
            action_surface_runtime_initializer
        ),
        world_runtime_provider=world_runtime_provider,
        actor_world_runtime_provider=actor_world_runtime_provider,
        explosion_candidate_provider=(
            region_arsenal_entity_only_explosion_candidates
            if action_surface_config is not None
            else None
        ),
        action_surface_provider=action_surface_provider,
        action_surface_executor=action_surface_executor,
        inventory_reset_provider=inventory_reset_provider,
        metadata=metadata,
    )


def load_distributed_arsenal_region_fixture(
    *,
    region_manifest: Path,
    traversal_manifest: Path,
    reset_pool_report: Path,
    expected_region_sha256: str,
    expected_traversal_sha256: str,
    expected_reset_pool_sha256: str,
    expected_reset_pool_file_sha256: str,
    working_set_capacity: int,
    batch_size: int,
    selection_key: int,
    assignment_key: int,
    params,
    runtime,
    action_surface_config: RegionActionSurfaceConfig | None = None,
    native_light_config: RegionNativeLightConfig | None = None,
) -> ArsenalRegionFixture:
    """Load a filtered multi-Region reset distribution without regeneration."""

    arguments = argparse.Namespace(
        region_manifest=region_manifest,
        traversal_manifest=traversal_manifest,
        reset_pool_report=reset_pool_report,
        expected_region_sha256=expected_region_sha256,
        expected_traversal_sha256=expected_traversal_sha256,
        expected_reset_pool_sha256=expected_reset_pool_sha256,
        expected_reset_pool_file_sha256=expected_reset_pool_file_sha256,
        working_set_capacity=working_set_capacity,
        batch_size=batch_size,
        region_selection_key=selection_key,
        region_assignment_key=assignment_key,
        region_action_surface_config=action_surface_config,
        region_native_light_config=native_light_config,
    )
    (
        geometry_provider,
        world_capability_provider,
        world_token_provider,
        reset_provider,
        metadata,
        action_surface_runtime_initializer,
        world_runtime_provider,
        actor_world_runtime_provider,
        action_surface_provider,
        action_surface_executor,
        inventory_reset_provider,
    ) = _distributed_region_case_with_geometry(arguments, params, runtime)
    return ArsenalRegionFixture(
        params=params,
        geometry_provider=geometry_provider,
        target_navigation_provider=(
            surrogate_region_walk_target_navigation_provider
        ),
        target_navigation_contract_sha256=(
            surrogate_region_walk_navigation_contract_sha256()
        ),
        world_capability_provider=world_capability_provider,
        world_token_provider=world_token_provider,
        reset_provider=reset_provider,
        action_surface_runtime_initializer=(
            action_surface_runtime_initializer
        ),
        world_runtime_provider=world_runtime_provider,
        actor_world_runtime_provider=actor_world_runtime_provider,
        explosion_candidate_provider=(
            region_arsenal_entity_only_explosion_candidates
            if action_surface_config is not None
            else None
        ),
        action_surface_provider=action_surface_provider,
        action_surface_executor=action_surface_executor,
        inventory_reset_provider=inventory_reset_provider,
        metadata=metadata,
    )


def exact_empty_local_geometry(params) -> GeometryState:
    """Return complete empty local coverage for exact-query throughput."""

    return GeometryState(
        origin=jnp.asarray(((0, 66, -1),), dtype=jnp.int32),
        cell_mask=jnp.ones((1, CELL_COUNT), dtype=jnp.bool_),
        flags=jnp.zeros((1, CELL_COUNT), dtype=jnp.int32),
        fluid_level=jnp.zeros((1, CELL_COUNT), dtype=jnp.int32),
        support=jnp.zeros((1, CELL_COUNT), dtype=jnp.int32),
        block_damage=jnp.zeros((1, CELL_COUNT), dtype=jnp.int32),
        fluid_damage=jnp.zeros((1, CELL_COUNT), dtype=jnp.int32),
        movement=jnp.zeros(
            (1, CELL_COUNT, MOVEMENT_FEATURES),
            dtype=jnp.float32,
        ),
        fluid_movement=jnp.zeros(
            (1, CELL_COUNT, FLUID_MOVEMENT_FEATURES),
            dtype=jnp.float32,
        ),
        collision_shape_index=jnp.full(
            (1, CELL_COUNT),
            COLLISION_SHAPE_NONE,
            dtype=jnp.int16,
        ),
        collision_full_cube_cell=jnp.full((1, 1), -1, dtype=jnp.int16),
        collision_exception_cell=jnp.full((1, 1), -1, dtype=jnp.int16),
        collision_exception_box_index=jnp.full((1, 1, 1), -1, dtype=jnp.int16),
        collision_exception_boxes=jnp.zeros((1, 1, 6), dtype=jnp.float32),
        collision_exception_box_cell=jnp.full((1, 1), -1, dtype=jnp.int16),
        collision_world_order=jnp.asarray(((0, 1),), dtype=jnp.int16),
        agent_bounds=params.agent_bounds[None, :],
        target_bounds=params.target_bounds[None, :],
        agent_los_offset=params.agent_eye_offset[None, :],
        target_los_offset=params.target_eye_offset[None, :],
    )


def _flat_benchmark_cases(params, config):
    """Return the control and exact-geometry benchmark bindings.

    Capability providers answer combat queries, while ``geometry_provider``
    is the physical motion/LOS input.  Exact cases must bind both; otherwise a
    benchmark can carry exact capability data while movement still runs on the
    flat fallback.
    """

    geometry = exact_empty_local_geometry(params)
    exact_provider = partial(
        geometry_arsenal_world_capabilities,
        geometry=geometry,
        config=config,
    )
    token_provider = partial(
        geometry_arsenal_world_tokens,
        geometry=geometry,
        role_opaque_mask=(geometry.flags[0] & FLAG_OPAQUE) != 0,
        geometry_provenance=WORLD_TOKEN_PROVENANCE_SURROGATE_EXACT_GEOMETRY,
        policy_config=DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
    )
    return (
        (
            "open_flat_control",
            None,
            open_flat_arsenal_world_capabilities,
            None,
            False,
            False,
            None,
            None,
            None,
            None,
            params,
        ),
        (
            "exact_geometry_without_tokens",
            geometry,
            exact_provider,
            None,
            False,
            False,
            None,
            None,
            None,
            None,
            params,
        ),
        (
            "exact_geometry_with_actor_tokens",
            geometry,
            exact_provider,
            token_provider,
            False,
            False,
            None,
            None,
            None,
            None,
            params,
        ),
    )


def _target_separation_range(
    value: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("target separation range must contain minimum and maximum")
    minimum, maximum = (float(item) for item in value)
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum <= 0.0
        or maximum <= minimum
    ):
        raise ValueError("target separation range must be finite, positive, and ordered")
    return minimum, maximum


def run_case(
    *,
    name,
    provider,
    token_provider,
    geometry_provider=None,
    target_navigation_provider=None,
    action_surface_provider=None,
    action_surface_executor=None,
    action_surface_runtime_initializer=None,
    world_runtime_provider=None,
    explosion_candidate_provider=None,
    reset_provider=None,
    inventory_reset_provider=None,
    params,
    config,
    batch,
    steps,
    repeats,
    seed,
    require_world_tokens=False,
    metadata=None,
):
    world_arguments = (
        {
            "world_capability_provider": provider,
            "world_token_provider": token_provider,
        }
        if world_runtime_provider is None
        else {
            "action_surface_runtime_initializer": (
                action_surface_runtime_initializer
            ),
            "world_runtime_provider": world_runtime_provider,
        }
    )
    if action_surface_provider is not None:
        world_arguments["action_surface_provider"] = action_surface_provider
    if action_surface_executor is not None:
        world_arguments["action_surface_executor"] = action_surface_executor
    if explosion_candidate_provider is not None:
        world_arguments["explosion_candidate_provider"] = (
            explosion_candidate_provider
        )
    if reset_provider is not None:
        world_arguments["reset_provider"] = reset_provider
    if inventory_reset_provider is None:
        initial_inventory = training_inventory_from_config(config)

        def inventory_reset_provider(keys, runtime_config):
            del keys, runtime_config
            return initial_inventory

    world_arguments["inventory_reset_provider"] = inventory_reset_provider
    environment = make_arsenal_environment(
        params,
        config,
        maximum_turn_degrees=REFERENCE_SKILL_MAXIMUM_TURN_DEGREES,
        geometry_provider=geometry_provider,
        target_navigation_provider=target_navigation_provider,
        **world_arguments,
    )

    def policy_observation(state, structured):
        return arsenal_policy_observation(
            structured,
            state.action_surface.block_candidates,
            state.action_surface.recipe_encoding,
            inventory_tokens=state.inventory_tokens,
            light_tokens=state.light_tokens,
        )

    def reset(keys):
        state, structured = environment.reset(keys)
        world = structured.world_geometry
        return (
            state,
            structured,
            policy_observation(state, structured),
            world.available,
            world.token_mask,
        )

    reset_keys = jax.random.split(jax.random.key(seed), batch)
    (
        state,
        structured,
        observation,
        reset_world_available,
        reset_world_mask,
    ) = jax.jit(reset)(reset_keys)
    jax.block_until_ready(observation)
    reset_available = int(jnp.count_nonzero(reset_world_available))
    reset_tokens = int(jnp.count_nonzero(reset_world_mask))
    reset_geometry_exhausted = int(
        jnp.count_nonzero(state.runtime.combat.geometry_exhausted)
    )
    reset_agent_grounded = int(jnp.count_nonzero(state.runtime.combat.agent_grounded))
    if require_world_tokens and (reset_available != batch or reset_tokens == 0):
        raise RuntimeError("Region traversal did not reach the policy observation")
    action = neutral_arsenal_policy_action_factors(batch)
    step_keys = jax.random.split(
        jax.random.key(seed + 1),
        steps * batch,
    ).reshape(steps, batch)

    def rollout(initial, keys):
        empty_structured = jax.tree.map(
            lambda value: jnp.zeros(value.shape, dtype=value.dtype),
            structured,
        )

        def body(carry, key):
            current, _ = carry
            next_state, structured, _, _, _ = environment.step_factors(
                current,
                action,
                key,
            )
            return (next_state, structured), ()

        (final_state, final_structured), _ = jax.lax.scan(
            body,
            (initial, empty_structured),
            keys,
        )
        world = final_structured.world_geometry
        return (
            final_state,
            policy_observation(final_state, final_structured),
            world.available,
            world.token_mask,
        )

    started = time.perf_counter()
    executable = jax.jit(rollout).lower(state, step_keys).compile()
    rollout_output = executable(state, step_keys)
    jax.block_until_ready(rollout_output)
    compile_seconds = time.perf_counter() - started
    memory = executable.memory_analysis()

    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        rollout_output = executable(state, step_keys)
        jax.block_until_ready(rollout_output)
        timings.append(time.perf_counter() - started)

    best_seconds = min(timings)
    median_seconds = float(np.median(timings))
    microticks = int(np.asarray(params.microticks))
    ticks_per_second = float(np.asarray(params.ticks_per_second))
    best_transitions_per_second = batch * steps / best_seconds
    median_transitions_per_second = batch * steps / median_seconds
    final_state, _, final_world_available, final_world_mask = rollout_output
    result = {
        "provider": name,
        "physical_geometry_bound": geometry_provider is not None,
        "physical_geometry_type": (
            None if geometry_provider is None else type(geometry_provider).__name__
        ),
        "target_navigation_bound": target_navigation_provider is not None,
        "target_navigation_contract_sha256": (
            surrogate_region_walk_navigation_contract_sha256()
            if target_navigation_provider is not None
            else None
        ),
        "unified_world_runtime_bound": world_runtime_provider is not None,
        "batch": batch,
        "steps": steps,
        "transitions": batch * steps,
        "microticks_per_transition": microticks,
        "native_ticks_per_second": ticks_per_second,
        "policy_observation_size": observation.shape[1],
        "rollout_observation_storage": "final_only",
        "compile_seconds": compile_seconds,
        "executable_memory_bytes": {
            name: (
                None
                if (value := getattr(memory, attribute, None)) is None
                else int(value)
            )
            for name, attribute in (
                ("arguments", "argument_size_in_bytes"),
                ("outputs", "output_size_in_bytes"),
                ("aliases", "alias_size_in_bytes"),
                ("temporaries", "temp_size_in_bytes"),
            )
        },
        "repeat_seconds": timings,
        "best_transitions_per_second": best_transitions_per_second,
        "median_transitions_per_second": median_transitions_per_second,
        "best_simulated_ticks_per_second": (
            best_transitions_per_second * microticks
        ),
        "median_simulated_ticks_per_second": (
            median_transitions_per_second * microticks
        ),
        "best_native_realtime_factor": (
            best_transitions_per_second * microticks / ticks_per_second
        ),
        "median_native_realtime_factor": (
            median_transitions_per_second * microticks / ticks_per_second
        ),
        "reset_world_geometry_available_rows": reset_available,
        "reset_world_geometry_tokens": reset_tokens,
        "reset_geometry_exhausted_rows": reset_geometry_exhausted,
        "reset_agent_grounded_rows": reset_agent_grounded,
        "reset_target_navigation_unsupported_rows": int(
            jnp.count_nonzero(
                state.runtime.combat.target_navigation_unsupported
            )
        ),
        "final_world_geometry_available_rows": int(
            jnp.count_nonzero(final_world_available)
        ),
        "final_world_geometry_tokens": int(
            jnp.count_nonzero(final_world_mask)
        ),
        "final_geometry_exhausted_rows": int(
            jnp.count_nonzero(final_state.runtime.combat.geometry_exhausted)
        ),
        "final_agent_grounded_rows": int(
            jnp.count_nonzero(final_state.runtime.combat.agent_grounded)
        ),
        "final_target_navigation_unsupported_rows": int(
            jnp.count_nonzero(
                final_state.runtime.combat.target_navigation_unsupported
            )
        ),
        "device": str(jax.devices()[0]),
    }
    if metadata is not None:
        result["world"] = metadata
    return result


def main() -> None:
    configure_persistent_compilation_cache(
        Path(__file__).resolve().parents[1] / ".codex-local" / "jax-cache"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=570057)
    parser.add_argument("--region-manifest", type=Path)
    parser.add_argument("--traversal-manifest", type=Path)
    parser.add_argument("--expected-region-sha256")
    parser.add_argument("--expected-traversal-sha256")
    parser.add_argument("--region-native-light-coverage-manifest", type=Path)
    parser.add_argument("--expected-region-native-light-coverage-sha256")
    parser.add_argument("--region-selection-key", type=int, default=1197923983)
    parser.add_argument("--region-assignment-key", type=int, default=1197923984)
    parser.add_argument("--region-node-seed", type=int, default=570057)
    parser.add_argument("--region-block-semantic-manifest", type=Path)
    parser.add_argument("--region-mutation-capacity", type=int)
    parser.add_argument("--block-action-maximum-distance", type=float)
    parser.add_argument("--block-action-view-sector-degrees", type=float)
    parser.add_argument(
        "--block-candidate-selection",
        choices=("native_camera", "dense_complete"),
        default="native_camera",
        help=(
            "native_camera matches player targeting; dense_complete is an "
            "explicit exhaustive mechanics/debug surface"
        ),
    )
    parser.add_argument("--native-item-interaction-recording", type=Path)
    parser.add_argument("--native-crafting-catalog-recording", type=Path)
    parser.add_argument(
        "--hytale-assets",
        type=Path,
        help="optional Assets.zip path for host-side block-action compilation",
    )
    parser.add_argument(
        "--enable-region-block-execution",
        action="store_true",
        help=(
            "enable source-checked exact Region block mutation; requires "
            "native item-interaction evidence"
        ),
    )
    parser.add_argument(
        "--region-held-item-asset-id",
        help=(
            "equip one native-evidence-backed item in actor hotbar slot zero "
            "for an explicit World-action scenario"
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help=(
            "run only the named benchmark case; repeat for multiple cases "
            "(the report remains explicitly partial)"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.repeats) < 1:
        parser.error("batch size, steps, and repeats must be positive")
    region_arguments = (
        args.region_manifest,
        args.traversal_manifest,
        args.expected_region_sha256,
        args.expected_traversal_sha256,
    )
    if any(value is not None for value in region_arguments) and not all(
        value is not None for value in region_arguments
    ):
        parser.error("all Region manifest and SHA-256 arguments are required together")
    native_light_arguments = (
        args.region_native_light_coverage_manifest,
        args.expected_region_native_light_coverage_sha256,
    )
    if any(value is not None for value in native_light_arguments) and not all(
        value is not None for value in native_light_arguments
    ):
        parser.error(
            "the Region native-light coverage manifest and SHA-256 are "
            "required together"
        )
    if all(value is not None for value in native_light_arguments) and (
        args.region_manifest is None
    ):
        parser.error("Region native light requires exact Region inputs")
    action_surface_arguments = (
        args.region_block_semantic_manifest,
        args.region_mutation_capacity,
        args.block_action_maximum_distance,
        args.block_action_view_sector_degrees,
    )
    if any(value is not None for value in action_surface_arguments) and not all(
        value is not None for value in action_surface_arguments
    ):
        parser.error(
            "all Region action-surface arguments are required together"
        )
    if (
        args.native_item_interaction_recording is not None
        and not all(value is not None for value in action_surface_arguments)
    ):
        parser.error(
            "native item evidence requires the Region action runtime"
        )
    if (
        args.enable_region_block_execution
        and args.native_item_interaction_recording is None
    ):
        parser.error(
            "Region block execution requires native item evidence"
        )
    if (
        args.native_crafting_catalog_recording is not None
        and not args.enable_region_block_execution
    ):
        parser.error(
            "Region fieldcraft requires the unified Region executor"
        )
    if (
        args.region_held_item_asset_id is not None
        and not args.enable_region_block_execution
    ):
        parser.error(
            "a Region held item requires Region block execution"
        )
    if (
        all(value is not None for value in action_surface_arguments)
        and args.region_manifest is None
    ):
        parser.error("Region action evidence requires exact Region inputs")
    args.region_action_surface_config = (
        None
        if not all(value is not None for value in action_surface_arguments)
        else RegionActionSurfaceConfig(
            block_semantic_manifest=args.region_block_semantic_manifest,
            mutation_capacity=args.region_mutation_capacity,
            maximum_distance=args.block_action_maximum_distance,
            view_sector_full_angle_degrees=(
                args.block_action_view_sector_degrees
            ),
            candidate_selection=args.block_candidate_selection,
            native_item_interaction_recording=(
                args.native_item_interaction_recording
            ),
            native_crafting_catalog_recording=(
                args.native_crafting_catalog_recording
            ),
            assets_path=args.hytale_assets,
            modification_allowed=args.enable_region_block_execution,
            held_item_asset_id=args.region_held_item_asset_id,
        )
    )
    args.region_native_light_config = (
        None
        if not all(value is not None for value in native_light_arguments)
        else RegionNativeLightConfig(
            coverage_manifest=args.region_native_light_coverage_manifest,
            expected_coverage_semantic_sha256=(
                args.expected_region_native_light_coverage_sha256
            ),
        )
    )

    params = default_combat_params(microticks=1)
    profiles = tuple(
        PROFILE_NAMES[index % len(PROFILE_NAMES)] for index in range(args.batch_size)
    )
    config = arsenal_runtime_config(hytale_0_5_7_loadouts(profiles))
    providers = _flat_benchmark_cases(params, config)
    region_reset_provider = None
    region_action_surface_executor = None
    region_inventory_reset_provider = None
    region_explosion_candidate_provider = None
    if args.region_manifest is not None:
        (
            region_params,
            region_geometry,
            region_provider,
            region_token_provider,
            region_metadata,
            region_runtime_initializer,
            region_world_runtime_provider,
            region_action_surface_provider,
            region_action_surface_executor,
            region_inventory_reset_provider,
        ) = _region_case_with_geometry(args, params, config)
        region_reset_provider = _fixed_region_reset_provider(
            agent_position=region_params.agent_spawn,
            target_position=(
                region_params.agent_spawn + region_params.target_offset
            ),
            batch_size=args.batch_size,
        )
        if args.region_action_surface_config is not None:
            region_explosion_candidate_provider = (
                region_arsenal_entity_only_explosion_candidates
            )
        providers += (
            (
                "native_region_navigation_unbound_control",
                region_geometry,
                region_provider,
                None,
                False,
                False,
                None,
                None,
                None,
                region_metadata,
                region_params,
            ),
            (
                "native_region_without_tokens",
                region_geometry,
                region_provider,
                None,
                False,
                True,
                None,
                None,
                None,
                region_metadata,
                region_params,
            ),
            (
                "native_region_with_actor_tokens",
                region_geometry,
                region_provider,
                region_token_provider,
                True,
                True,
                region_action_surface_provider,
                region_runtime_initializer,
                region_world_runtime_provider,
                region_metadata,
                region_params,
            ),
        )
    available_case_names = tuple(row[0] for row in providers)
    if args.cases:
        requested_case_names = tuple(dict.fromkeys(args.cases))
        unknown_case_names = tuple(
            name
            for name in requested_case_names
            if name not in available_case_names
        )
        if unknown_case_names:
            parser.error(
                "unknown or unavailable benchmark case(s): "
                + ", ".join(unknown_case_names)
            )
        requested_case_set = frozenset(requested_case_names)
        providers = tuple(
            row for row in providers if row[0] in requested_case_set
        )
    results = []
    for (
        name,
        geometry_provider,
        provider,
        token_provider,
        required,
        bind_navigation,
        action_surface_provider,
        action_surface_runtime_initializer,
        world_runtime_provider,
        metadata,
        case_params,
    ) in providers:
        row = run_case(
            name=name,
            geometry_provider=geometry_provider,
            target_navigation_provider=(
                surrogate_region_walk_target_navigation_provider
                if bind_navigation
                else None
            ),
            provider=provider,
            token_provider=token_provider,
            action_surface_provider=action_surface_provider,
            action_surface_executor=(
                region_action_surface_executor
                if action_surface_provider is not None
                else None
            ),
            action_surface_runtime_initializer=(
                action_surface_runtime_initializer
            ),
            world_runtime_provider=world_runtime_provider,
            explosion_candidate_provider=(
                region_explosion_candidate_provider
                if name.startswith("native_region_")
                else None
            ),
            reset_provider=(
                region_reset_provider
                if name.startswith("native_region_")
                else None
            ),
            inventory_reset_provider=(
                region_inventory_reset_provider
                if action_surface_provider is not None
                else None
            ),
            params=case_params,
            config=config,
            batch=args.batch_size,
            steps=args.steps,
            repeats=args.repeats,
            seed=args.seed,
            require_world_tokens=required,
            metadata=metadata,
        )
        results.append(row)
        print(json.dumps(row), flush=True)
    if results[0]["policy_observation_size"] != ARSENAL_POLICY_OBSERVATION_SIZE:
        raise RuntimeError("benchmark policy observation contract drift")
    report = {
        "schema": _BENCHMARK_SCHEMA,
        "seed": args.seed,
        "world_geometry_token_contract_sha256": (
            world_geometry_token_contract_sha256()
        ),
        "arsenal_policy_contract_sha256": (
            arsenal_policy_contract_sha256(DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG)
        ),
        "canonical_native_evidence_bridge_sha256": (
            current_native_evidence_jar_sha256()
        ),
        "case_selection": "explicit" if args.cases else "all",
        "available_cases": list(available_case_names),
        "selected_cases": [row["provider"] for row in results],
        "complete_case_set": len(results) == len(available_case_names),
        "results": results,
    }
    report["report_semantic_sha256"] = hashlib.sha256(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
    print(json.dumps(report, indent=2), flush=True)


def _region_case(args, params, config):
    """Build the historical capability/token Region tuple for profile tools."""

    (
        region_params,
        _,
        provider,
        token_provider,
        metadata,
        _,
        _,
        _,
        _,
        _,
    ) = _region_case_with_geometry(args, params, config)
    return region_params, provider, token_provider, metadata


def _region_world_bindings(
    selection,
    params,
    config,
    *,
    region_manifest_path: Path,
    action_surface_config: RegionActionSurfaceConfig | None = None,
    native_light_config: RegionNativeLightConfig | None = None,
):
    """Bind one Region selection to physical, capability, and token seams."""

    traversal_atlas = selection.traversal_atlas
    if traversal_atlas is None:
        raise RuntimeError("Region selection lacks its traversal graph")
    base_geometry = region_geometry_from_atlas(
        selection.atlas,
        selection.environment_world_id,
        agent_bounds=params.agent_bounds,
        target_bounds=params.target_bounds,
        agent_los_offset=params.agent_eye_offset,
        target_los_offset=params.target_eye_offset,
        # Perception LOS is budgeted by how far an actor can see, not by how far
        # it can reach a block. The block reach (4.0) sized the step buffer to 10
        # cells, so every entity ray past ~9 blocks reported `capacity_exceeded`
        # and read as "not visible" -- indistinguishable from real occlusion.
        # A segment of length D crosses at most ceil(sqrt(3)*D)+2 cells, so 72
        # blocks is a 128-cell budget: well clear of sensor_range (16) and the
        # 20-block proximity radius the pursuit lesson scores against.
        max_los_distance=REGION_PERCEPTION_LOS_DISTANCE_BLOCKS,
        # Match `arena/worlds.py`'s `_COMBAT_SWEEP_CELL_CAPACITY`. That path
        # raised the swept-AABB working set to four rows per axis because a
        # 1.805-block actor plus one 30 Hz fall/knockback step touches four,
        # and recorded that three "falsely exhausted valid Region motion
        # mid-duel". This constructor never got the same argument, so every
        # fixture built here -- including every pursuit training run -- ran at
        # the `_LOCAL_SWEEP_SIZE = 3` default while its sibling ran at 4.
        #
        # Measured on this atlas, exhaustion is a step-size cliff: at capacity
        # 3 a 1.25-block step exhausts 98.4% of lanes and 1.5 blocks exhausts
        # 100%, while capacity 4 is clean to 2.0. Since `geometry_exhausted`
        # is sticky and truncates the episode, one oversized fall step ends a
        # run permanently.
        sweep_cell_capacity=REGION_COMBAT_SWEEP_CELL_CAPACITY,
    )
    role_mask = (
        selection.atlas.cell_flags.astype(jnp.int32) & jnp.int32(FLAG_OPAQUE)
    ) != 0
    initial_world_runtime: object = base_geometry
    action_surface_provider = None
    action_surface_executor = None
    inventory_reset_provider = None
    if action_surface_config is not None:
        initial_world_runtime = region_action_runtime_from_artifact_libraries(
            selection,
            base_geometry,
            region_manifest_path=region_manifest_path,
            block_semantic_manifest_path=(
                action_surface_config.block_semantic_manifest
            ),
            mutation_capacity=action_surface_config.mutation_capacity,
        )
        if action_surface_config.candidate_selection == "native_camera":
            block_provider = make_region_runtime_camera_block_candidate_provider(
                maximum_interaction_distance=(
                    action_surface_config.maximum_distance
                ),
            )
        else:
            block_provider = make_region_runtime_block_candidate_provider(
                role_opaque_mask=role_mask,
                maximum_distance=action_surface_config.maximum_distance,
                view_sector_full_angle_radians=np.deg2rad(
                    action_surface_config.view_sector_full_angle_degrees
                ),
                cell_radius=math.ceil(action_surface_config.maximum_distance),
            )
        item_provider = None
        recipe_provider = None
        crafting_assets = None
        if action_surface_config.native_crafting_catalog_recording is not None:
            crafting_assets = compile_region_fieldcraft_assets(
                action_surface_config.native_crafting_catalog_recording,
                config.inventory_layout,
                expected_bridge_sha256=current_native_evidence_jar_sha256(),
                assets_path=action_surface_config.assets_path,
            )
            recipe_provider = make_region_fieldcraft_candidate_provider(
                crafting_assets,
                block_candidate_provider=block_provider,
            )
        if action_surface_config.native_item_interaction_recording is not None:
            recording = json.loads(
                action_surface_config.native_item_interaction_recording.read_text(
                    encoding="utf-8"
                )
            )
            item_evidence = item_interaction_evidence_from_native_recording(
                recording,
                expected_bridge_sha256=(
                    current_native_evidence_jar_sha256()
                ),
            )
            item_provider = make_inventory_item_interaction_availability_provider(
                item_interaction_table_from_native_evidence(
                    item_evidence,
                    expected_bridge_sha256=(
                        current_native_evidence_jar_sha256()
                    ),
                ),
                config.inventory_layout,
            )
        if (
            action_surface_config.modification_allowed
            and action_surface_config.native_item_interaction_recording
            is not None
        ):
            executor_assets = compile_region_block_executor_assets(
                selection,
                initial_world_runtime,
                item_evidence,
                region_manifest_path=region_manifest_path,
                block_semantic_manifest_path=(
                    action_surface_config.block_semantic_manifest
                ),
                assets_path=action_surface_config.assets_path,
            )
            action_surface_executor = make_region_block_action_surface_executor(
                block_candidate_provider=block_provider,
                interaction_table=executor_assets.interactions,
                held_tool_table=executor_assets.held_tools,
                placed_geometry_table=executor_assets.placed_geometry,
                block_catalog=executor_assets.block_catalog,
                gather_defaults=executor_assets.gather_defaults,
                inventory_layout=config.inventory_layout,
                drop_programs=executor_assets.drop_programs,
                recipe_candidate_provider=recipe_provider,
                crafting_assets=crafting_assets,
                state_change_use_table=executor_assets.state_change_use,
                modification_allowed=True,
            )
            action_surface_executor.excluded_asset_references = (
                executor_assets.excluded_asset_references
            )
            item_provider = make_region_block_item_availability_provider(
                executor_assets.interactions,
                config.inventory_layout,
                block_candidate_provider=block_provider,
                state_change_use_table=executor_assets.state_change_use,
            )
            if action_surface_config.held_item_asset_id is not None:
                inventory_reset_provider = (
                    make_region_held_item_inventory_reset_provider(
                        executor_assets.interactions,
                        config.inventory_layout,
                        action_surface_config.held_item_asset_id,
                    )
                )
        action_surface_provider = make_arsenal_action_surface_provider(
            block_candidate_provider=block_provider,
            recipe_candidate_provider=recipe_provider,
            item_interaction_availability_provider=item_provider,
        )
    geometry = (
        initial_world_runtime.geometry
        if isinstance(initial_world_runtime, RegionActionRuntimeState)
        else initial_world_runtime
    )
    provider = partial(
        geometry_arsenal_world_capabilities,
        geometry=geometry,
        config=config,
    )
    region_surrogate_light_atlas = region_surrogate_sky_light_atlas_from_region(
        selection.atlas
    )
    region_native_light_atlas = (
        None
        if native_light_config is None
        else region_native_light_atlas_from_coverage_library(
            selection,
            region_manifest_path=region_manifest_path,
            light_coverage_manifest_path=(
                native_light_config.coverage_manifest
            ),
            expected_light_coverage_semantic_sha256=(
                native_light_config.expected_coverage_semantic_sha256
            ),
        )
    )

    def token_source_for_geometry(state, combat_params, current_geometry):
        actor_position = state.combat.position[:, :1, :]
        actor_eye_position = (
            actor_position
            + combat_params.agent_eye_offset[None, None, :]
        )
        yaw_radians = jnp.deg2rad(state.combat.yaw[:, 0])
        actor_forward = jnp.stack(
            (
                -jnp.sin(yaw_radians),
                jnp.zeros_like(yaw_radians),
                -jnp.cos(yaw_radians),
            ),
            axis=1,
        )[:, None, :]
        return produce_actor_region_world_geometry_tokens(
            traversal_atlas,
            selection.environment_world_id,
            current_geometry,
            actor_position,
            actor_eye_position,
            actor_forward,
            role_opaque_mask=role_mask,
            geometry_provenance=(
                WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY
            ),
            traversal_staging_capacity=(
                REGION_TRAVERSAL_POLICY_STAGING_CAPACITY
            ),
            token_capacity=(
                DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG.token_capacity
            ),
            maximum_distance=(
                DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG.maximum_distance
            ),
        )
    def token_provider_for_geometry(state, combat_params, current_geometry):
        return encode_world_geometry_policy_tokens(
            token_source_for_geometry(state, combat_params, current_geometry),
            actor_index=0,
            config=DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
        )

    def token_provider(state, combat_params):
        return token_provider_for_geometry(state, combat_params, geometry)

    def initialize_world_runtime(_keys, _state, _params, _config):
        return initial_world_runtime

    def project_world_runtime(
        state,
        current_world,
        combat_params,
        runtime_config,
    ):
        current_geometry = (
            current_world.geometry
            if isinstance(current_world, RegionActionRuntimeState)
            else current_world
        )
        source_tokens = token_source_for_geometry(
            state,
            combat_params,
            current_geometry,
        )
        actor_position = state.combat.position[:, :1, :]
        surrogate_light = region_surrogate_actor_block_light_tokens(
            region_surrogate_light_atlas,
            current_geometry,
            source_tokens,
            actor_position,
        )
        selected_light = surrogate_light
        if region_native_light_atlas is not None:
            selected_light = native_or_surrogate_actor_block_light_tokens(
                region_native_actor_block_light_tokens(
                    region_native_light_atlas,
                    current_geometry,
                    source_tokens,
                    actor_position,
                ),
                surrogate_light,
            )
        return ArsenalWorldRuntimeViews(
            physical_geometry=current_geometry,
            capabilities=geometry_arsenal_world_capabilities(
                state,
                combat_params,
                geometry=current_geometry,
                config=runtime_config,
            ),
            features=empty_injected_world_features(
                state.combat.health.shape[0]
            ),
            tokens=encode_world_geometry_policy_tokens(
                source_tokens,
                actor_index=0,
                config=DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
            ),
            light_tokens=encode_actor_light_policy_tokens(
                selected_light,
                actor_index=0,
            ),
        )

    def world_runtime_provider(state, current_world, combat_params):
        return project_world_runtime(
            state,
            current_world,
            combat_params,
            config,
        )

    def actor_world_runtime_provider(
        actor_state,
        current_world,
        combat_params,
        actor_config,
    ):
        return project_world_runtime(
            actor_state,
            current_world,
            combat_params,
            actor_config,
        )

    return (
        geometry,
        provider,
        token_provider,
        role_mask,
        initialize_world_runtime,
        world_runtime_provider,
        actor_world_runtime_provider,
        action_surface_provider,
        action_surface_executor,
        inventory_reset_provider,
    )


def _reachable_separated_nodes(
    source: int,
    positions: np.ndarray,
    node_mask: np.ndarray,
    edge_mask: np.ndarray,
    edge_destination: np.ndarray,
    destination_mask: np.ndarray,
    separation_range: tuple[float, float],
) -> np.ndarray:
    """Return graph-reachable target nodes in one horizontal distance band."""

    reachable = np.zeros(node_mask.shape, dtype=np.bool_)
    reachable[source] = True
    pending = [source]
    while pending:
        node = pending.pop()
        for raw in edge_destination[node, edge_mask[node]]:
            destination = int(raw)
            if (
                0 <= destination < node_mask.size
                and node_mask[destination]
                and not reachable[destination]
            ):
                reachable[destination] = True
                pending.append(destination)
    distance = np.linalg.norm(
        positions[:, (0, 2)] - positions[source, (0, 2)], axis=1
    )
    minimum, maximum = separation_range
    return np.flatnonzero(
        reachable
        & destination_mask
        & (distance >= minimum)
        & (distance <= maximum)
    )


def _first_traversal_legal_region_pair(
    source_order,
    positions,
    node_mask,
    edge_mask,
    edge_destination,
    destination_mask,
    separation_range,
    random,
):
    """Choose a separated pair from the actor-profile traversal graph."""

    source_limit = min(len(source_order), 64)
    for source_rank, raw_source in enumerate(
        source_order[:source_limit], start=1
    ):
        source = int(raw_source)
        separated = _reachable_separated_nodes(
            source,
            positions,
            node_mask,
            edge_mask,
            edge_destination,
            destination_mask,
            separation_range,
        )
        shuffled = random.permutation(separated)
        if not shuffled.size:
            continue
        horizontal_distance = np.linalg.norm(
            positions[shuffled][:, (0, 2)] - positions[source, (0, 2)],
            axis=1,
        )
        vertical_distance = np.abs(
            positions[shuffled, 1] - positions[source, 1]
        )
        priority = np.lexsort(
            (
                np.arange(shuffled.size),
                horizontal_distance,
                vertical_distance,
            )
        )
        destination = int(shuffled[int(priority[0])])
        return source, destination, source_rank, 1
    raise RuntimeError(
        "selected Region has no actor-profile traversal pair in separation range "
        f"{separation_range} after {source_limit} sources"
    )


def _first_actor_legal_visible_region_pair(
    source_order,
    positions,
    node_mask,
    edge_mask,
    edge_destination,
    destination_mask,
    separation_range,
    traversal_atlas,
    environment_world_id,
    geometry,
    role_mask,
    params,
    random,
):
    """Select one reachable visible pair in bounded batched device checks."""

    batch = int(environment_world_id.shape[0])
    actor_legal, pair_visible = _region_pair_predicates(
        traversal_atlas,
        environment_world_id,
        geometry,
        role_mask,
        params,
    )

    # Region traversal graphs can contain tens of thousands of vertical nodes.
    # A random destination order wastes reset time testing lines through cliffs.
    # Prefer level, short pairs and bound the exact device checks; this keeps
    # selection deterministic without changing runtime geometry semantics.
    source_limit = min(len(source_order), 64)
    destination_limit = 64
    pending = []
    checked_pairs = 0
    visible_pairs = 0
    source_legal_pairs = 0
    destination_legal_pairs = 0

    def evaluate(rows):
        nonlocal checked_pairs, visible_pairs
        nonlocal source_legal_pairs, destination_legal_pairs
        count = len(rows)
        padded = rows + [rows[-1]] * (batch - count)
        source_position = jnp.asarray(
            [positions[source] for source, _, _, _ in padded],
            dtype=jnp.float32,
        )
        destination_position = jnp.asarray(
            [positions[destination] for _, destination, _, _ in padded],
            dtype=jnp.float32,
        )
        visible = np.asarray(
            jax.device_get(pair_visible(source_position, destination_position))
        )[:count]
        checked_pairs += count
        visible_pairs += int(np.count_nonzero(visible))
        if not np.any(visible):
            return None
        source_legal = np.asarray(
            jax.device_get(actor_legal(source_position))
        )[:count]
        destination_legal = np.asarray(
            jax.device_get(actor_legal(destination_position))
        )[:count]
        source_legal_pairs += int(np.count_nonzero(source_legal))
        destination_legal_pairs += int(np.count_nonzero(destination_legal))
        legal = visible & source_legal & destination_legal
        found = np.flatnonzero(legal)
        return None if not found.size else rows[int(found[0])]

    for source_rank, raw_source in enumerate(
        source_order[:source_limit], start=1
    ):
        source = int(raw_source)
        separated = _reachable_separated_nodes(
            source,
            positions,
            node_mask,
            edge_mask,
            edge_destination,
            destination_mask,
            separation_range,
        )
        shuffled = random.permutation(separated)
        if shuffled.size:
            horizontal_distance = np.linalg.norm(
                positions[shuffled][:, (0, 2)]
                - positions[source, (0, 2)],
                axis=1,
            )
            vertical_distance = np.abs(
                positions[shuffled, 1] - positions[source, 1]
            )
            priority = np.lexsort(
                (
                    np.arange(shuffled.size),
                    horizontal_distance,
                    vertical_distance,
                )
            )
            shuffled = shuffled[priority[:destination_limit]]
        for destination_rank, raw_destination in enumerate(shuffled, start=1):
            pending.append(
                (source, int(raw_destination), source_rank, destination_rank)
            )
            if len(pending) == batch:
                selected = evaluate(pending)
                if selected is not None:
                    return selected
                pending.clear()
    if pending:
        selected = evaluate(pending)
        if selected is not None:
            return selected
    raise RuntimeError(
        "selected Region has no reachable actor-legal pair with certified line of sight "
        f"in separation range {separation_range} after {source_limit} sources "
        f"and {destination_limit} ranked destinations per source "
        f"(checked={checked_pairs}, visible={visible_pairs}, "
        f"source_legal={source_legal_pairs}, "
        f"destination_legal={destination_legal_pairs})"
    )


def _region_pair_predicates(
    traversal_atlas,
    environment_world_id,
    geometry,
    role_mask,
    params,
):
    """Batched actor-legality and line-of-sight checks over a Region batch.

    Shared so the single-pair and per-environment selectors certify a spawn
    identically -- a row placed by one and validated by the other would make
    two "Region" resets incomparable. ``environment_world_id`` carries each
    row's world, so a batch spanning several worlds is checked in place.
    """

    batch = int(environment_world_id.shape[0])
    forward = jnp.broadcast_to(
        jnp.asarray((0.0, 0.0, -1.0), dtype=jnp.float32),
        (batch, 1, 3),
    )

    @jax.jit
    def actor_legal(actor_position):
        actor = actor_position[:, None, :]
        tokens = produce_actor_region_world_geometry_tokens(
            traversal_atlas,
            environment_world_id,
            geometry,
            actor,
            actor + params.agent_eye_offset[None, None, :],
            forward,
            role_opaque_mask=role_mask,
            geometry_provenance=(WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY),
            traversal_staging_capacity=(REGION_TRAVERSAL_POLICY_STAGING_CAPACITY),
            token_capacity=DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG.token_capacity,
            maximum_distance=(DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG.maximum_distance),
        )
        available = jnp.all(tokens.available.reshape((batch, -1)), axis=1)
        populated = jnp.any(tokens.token_mask.reshape((batch, -1)), axis=1)
        return available & populated

    @jax.jit
    def pair_visible(source_position, destination_position):
        line_of_sight = geometry_perception_line_of_sight_result(
            geometry,
            source_position + params.agent_eye_offset,
            destination_position + params.target_eye_offset,
            role_opaque_mask=None,
        )
        return (
            line_of_sight.visible
            & ~line_of_sight.geometry_exhausted
            & ~line_of_sight.capacity_exceeded
            & ~line_of_sight.invalid
        )

    return actor_legal, pair_visible


def _reachable_nodes_from(
    source: int,
    node_mask: np.ndarray,
    edge_mask: np.ndarray,
    edge_destination: np.ndarray,
) -> np.ndarray:
    """Forward-reachable node set, expanded a frontier at a time.

    Same traversal semantics as :func:`_reachable_separated_nodes`, but the
    per-node Python loop there costs seconds on a 40k-node graph. Per-row
    spawn diversity needs one flood per environment, so the frontier is
    expanded with array gathers instead.
    """

    reachable = np.zeros(node_mask.shape, dtype=np.bool_)
    reachable[source] = True
    frontier = np.asarray((source,), dtype=np.int64)
    node_count = int(node_mask.size)
    while frontier.size:
        destinations = edge_destination[frontier][edge_mask[frontier]]
        destinations = destinations[
            (destinations >= 0) & (destinations < node_count)
        ]
        if not destinations.size:
            break
        destinations = destinations[node_mask[destinations]]
        destinations = destinations[~reachable[destinations]]
        if not destinations.size:
            break
        frontier = np.unique(destinations)
        reachable[frontier] = True
    return reachable


def _per_environment_region_pairs(
    *,
    world_ids,
    candidates_by_world,
    positions_by_world,
    node_mask_by_world,
    edge_mask_by_world,
    edge_destination_by_world,
    destination_mask_by_world,
    separation_range,
    actor_legal,
    pair_visible,
    random,
    source_rounds: int = 48,
    destination_rounds: int = 512,
):
    """Sample one qualifying (source, destination) pair per environment row.

    The single-pair selector returns the first legal pair and broadcasts it to
    every row, so a batch is byte-identical and the policy can memorise one
    spawn. Here each row draws its own pair *from its own world*, and because
    ``environment_world_id`` already carries each row's world the device
    predicates check the whole batch in place -- one round costs the same three
    calls the single-pair search already paid.

    Sources and destinations are resolved in two phases on purpose. Actor
    legality is a property of the *node*, so an illegal source fails every
    destination hung off it: a first cut that shared 6 sources per world left 4
    of 16 rows unplaceable with no certified pair at all, while re-drawing
    sources freely placed all but one. Phase one therefore spends its device
    calls finding legal sources, and phase two reuses them.
    """

    batch = int(world_ids.shape[0])
    minimum, maximum = separation_range
    worlds = sorted({int(world) for world in world_ids})
    for world in worlds:
        if not candidates_by_world[world].size:
            raise RuntimeError(
                f"Region world {world} has no core traversal edge"
            )

    def positions_for(rows_source):
        return jnp.asarray(
            np.stack(
                [
                    positions_by_world[int(world_ids[row])][rows_source[row]]
                    for row in range(batch)
                ]
            ),
            dtype=jnp.float32,
        )

    # Phase 1: collect actor-legal source nodes per world. Every row proposes a
    # fresh candidate from its own world each round, so one device call tests
    # one candidate per resident world.
    legal_sources: dict[int, list[int]] = {world: [] for world in worlds}
    probe = np.zeros((batch,), dtype=np.int64)
    for _ in range(source_rounds):
        if all(len(legal_sources[world]) >= 4 for world in worlds):
            break
        for row in range(batch):
            candidates = candidates_by_world[int(world_ids[row])]
            probe[row] = int(random.choice(candidates))
        verdict = np.asarray(jax.device_get(actor_legal(positions_for(probe))))
        for row in np.flatnonzero(verdict):
            world = int(world_ids[row])
            if len(legal_sources[world]) < 8:
                legal_sources[world].append(int(probe[row]))
    starved = [world for world in worlds if not legal_sources[world]]
    if starved:
        raise RuntimeError(
            "per-environment Region reset found no actor-legal source in "
            f"{len(starved)} of {len(worlds)} resident worlds after "
            f"{source_rounds} rounds (worlds {starved})"
        )

    # Phase 2: hang ranked destinations off the legal sources. A uniformly
    # random destination in the band is usually a sightline through a cliff --
    # random order left 12 of 16 rows unplaced -- so rank level-first then
    # shortest, the same order the single-pair search uses.
    pools: dict[int, list[tuple[int, int]]] = {}
    for world in worlds:
        positions = positions_by_world[world]
        pairs: list[tuple[int, int]] = []
        for chosen in legal_sources[world]:
            reachable = _reachable_nodes_from(
                chosen,
                node_mask_by_world[world],
                edge_mask_by_world[world],
                edge_destination_by_world[world],
            )
            distance = np.linalg.norm(
                positions[:, (0, 2)] - positions[chosen, (0, 2)],
                axis=1,
            )
            separated = np.flatnonzero(
                reachable
                & destination_mask_by_world[world]
                & (distance >= minimum)
                & (distance <= maximum)
            )
            if not separated.size:
                continue
            shuffled = random.permutation(separated)
            horizontal = np.linalg.norm(
                positions[shuffled][:, (0, 2)] - positions[chosen, (0, 2)],
                axis=1,
            )
            vertical = np.abs(positions[shuffled, 1] - positions[chosen, 1])
            priority = np.lexsort(
                (np.arange(shuffled.size), horizontal, vertical)
            )
            pairs.extend(
                (chosen, int(node)) for node in shuffled[priority[:128]]
            )
        if not pairs:
            raise RuntimeError(
                f"Region world {world} has no graph-reachable node in "
                f"separation range {separation_range} from any of its "
                f"{len(legal_sources[world])} actor-legal sources"
            )
        pools[world] = pairs

    source = np.zeros((batch,), dtype=np.int64)
    destination = np.zeros((batch,), dtype=np.int64)
    filled = np.zeros((batch,), dtype=np.bool_)
    # Rows sharing a world walk the shared pool from staggered offsets so they
    # do not all propose the same pair on the same round.
    cursors = np.array(
        [(row * 7) % len(pools[int(world_ids[row])]) for row in range(batch)],
        dtype=np.int64,
    )
    validated: dict[int, list[tuple[int, int]]] = {}
    for _ in range(destination_rounds):
        pending = np.flatnonzero(~filled)
        if not pending.size:
            break
        for row in pending:
            pool = pools[int(world_ids[row])]
            source[row], destination[row] = pool[
                int(cursors[row]) % len(pool)
            ]
            cursors[row] += 1
        source_position = positions_for(source)
        destination_position = positions_for(destination)
        legal = (
            np.asarray(
                jax.device_get(
                    pair_visible(source_position, destination_position)
                )
            )
            & np.asarray(jax.device_get(actor_legal(destination_position)))
        )
        for row in np.flatnonzero(legal & ~filled):
            validated.setdefault(int(world_ids[row]), []).append(
                (int(source[row]), int(destination[row]))
            )
        filled |= legal

    # A world whose band is mostly blocked can starve one row while its
    # neighbours fill immediately. Reuse a pair already certified in that row's
    # own world rather than failing the build: the row stays legal and only its
    # uniqueness is given up.
    for row in np.flatnonzero(~filled):
        certified = validated.get(int(world_ids[row]))
        if certified:
            source[row], destination[row] = certified[
                int(row) % len(certified)
            ]
            filled[row] = True

    if not bool(np.all(filled)):
        unplaced = sorted(
            {int(world_ids[row]) for row in np.flatnonzero(~filled)}
        )
        raise RuntimeError(
            "per-environment Region reset could not place "
            f"{int(np.count_nonzero(~filled))} of {batch} rows with a "
            f"visible actor-legal pair in separation range {separation_range} "
            f"(worlds {unplaced})"
        )
    return source, destination


def _region_case_with_geometry(args, params, config):
    """Build one Region case including its fixed physical geometry provider.

    ``region_working_set_capacity`` publishes that many worlds and spreads the
    batch across them; ``region_environment_diversity`` then gives every row
    its own spawn/target pair drawn inside its own world. Both default to the
    historical single-world, single-pair, byte-identical batch so existing
    fixtures and their recorded observation pins are unchanged.
    """

    working_set_capacity = int(
        getattr(args, "region_working_set_capacity", 1) or 1
    )
    environment_diversity = bool(
        getattr(args, "region_environment_diversity", False)
    )
    region_manifest = json.loads(args.region_manifest.read_text(encoding="utf-8"))
    traversal_manifest = json.loads(args.traversal_manifest.read_text(encoding="utf-8"))
    selection = region_environments_from_artifact_manifest(
        args.region_manifest,
        usage=REGION_ARTIFACT_USAGE_TRAINING,
        expected_library_semantic_sha256=args.expected_region_sha256,
        working_set_capacity=working_set_capacity,
        environment_count=args.batch_size,
        selection_key=args.region_selection_key,
        assignment_key=args.region_assignment_key,
        selection_seeds=getattr(args, "region_selection_seeds", None),
        traversal_manifest_path=args.traversal_manifest,
        expected_traversal_library_semantic_sha256=(args.expected_traversal_sha256),
    )
    traversal_atlas = selection.traversal_atlas
    if traversal_atlas is None:
        raise RuntimeError("Region selection lacks its traversal graph")
    # The atlas leads with a world axis. World 0 is the only one the
    # single-pair path ever reads; the per-environment path indexes each row's
    # own world, so both views are derived from the same arrays here.
    node_mask_by_world = np.asarray(traversal_atlas.node_mask)
    edge_mask_by_world = np.asarray(traversal_atlas.edge_mask)
    positions_by_world = np.asarray(traversal_atlas.node_position)
    core_origin_by_world = (
        np.asarray(traversal_atlas.core_min_chunk_xz) * CHUNK_SIZE
    )
    candidates_by_world = []
    inside_core_by_world = []
    for world in range(node_mask_by_world.shape[0]):
        blocks_world = np.floor(positions_by_world[world]).astype(np.int32)
        origin = core_origin_by_world[world]
        inside = np.all(
            (blocks_world[:, (0, 2)] >= origin)
            & (blocks_world[:, (0, 2)] < origin + CORE_BLOCKS_PER_AXIS),
            axis=1,
        )
        inside_core_by_world.append(inside)
        candidates_by_world.append(
            np.flatnonzero(
                node_mask_by_world[world]
                & inside
                & np.any(edge_mask_by_world[world], axis=1)
            )
        )
    node_mask = node_mask_by_world[0]
    edge_mask = edge_mask_by_world[0]
    positions = positions_by_world[0]
    inside_core = inside_core_by_world[0]
    candidates = candidates_by_world[0]
    if not candidates.size:
        raise RuntimeError("selected Region graph has no core traversal edge")
    (
        geometry,
        provider,
        token_provider,
        role_mask,
        action_surface_runtime_initializer,
        world_runtime_provider,
        actor_world_runtime_provider,
        action_surface_provider,
        action_surface_executor,
        inventory_reset_provider,
    ) = _region_world_bindings(
        selection,
        params,
        config,
        region_manifest_path=args.region_manifest,
        action_surface_config=getattr(
            args,
            "region_action_surface_config",
            None,
        ),
        native_light_config=getattr(
            args,
            "region_native_light_config",
            None,
        ),
    )
    random = np.random.default_rng(args.region_node_seed)
    first_source = int(random.choice(candidates))
    remaining_sources = candidates[candidates != first_source]
    source_order = np.concatenate(
        (
            np.asarray((first_source,), dtype=candidates.dtype),
            random.permutation(remaining_sources),
        )
    )
    requested_separation = getattr(args, "region_target_separation_range", None)
    require_initial_los = bool(
        getattr(args, "region_require_initial_line_of_sight", True)
    )
    environment_agent_position = None
    environment_target_position = None
    if environment_diversity:
        if requested_separation is None or not require_initial_los:
            raise ValueError(
                "per-environment Region spawns require a target separation "
                "range and an initial line-of-sight requirement"
            )
        world_ids = np.asarray(selection.environment_world_id)
        actor_legal, pair_visible = _region_pair_predicates(
            traversal_atlas,
            selection.environment_world_id,
            geometry,
            role_mask,
            params,
        )
        environment_source, environment_destination = (
            _per_environment_region_pairs(
                world_ids=world_ids,
                candidates_by_world=candidates_by_world,
                positions_by_world=positions_by_world,
                node_mask_by_world=node_mask_by_world,
                edge_mask_by_world=edge_mask_by_world,
                edge_destination_by_world=(
                    np.asarray(traversal_atlas.edge_destination)
                ),
                destination_mask_by_world=inside_core_by_world,
                separation_range=requested_separation,
                actor_legal=actor_legal,
                pair_visible=pair_visible,
                random=random,
            )
        )
        environment_agent_position = np.stack(
            [
                positions_by_world[int(world_ids[row])][
                    environment_source[row]
                ]
                for row in range(world_ids.shape[0])
            ]
        )
        environment_target_position = np.stack(
            [
                positions_by_world[int(world_ids[row])][
                    environment_destination[row]
                ]
                for row in range(world_ids.shape[0])
            ]
        )
        # Row 0 still seeds the scalar params so anything reading
        # ``agent_spawn``/``floor_y`` as a scalar keeps a real, legal spawn.
        source = int(environment_source[0])
        destination = int(environment_destination[0])
        source_rank = 1
        destination_rank = 1
        destination_method = (
            "per_environment_graph_reachable_actor_legal_visible_pair_v1"
        )
        positions = positions_by_world[int(world_ids[0])]
    elif requested_separation is None:
        source, source_rank = _first_actor_legal_region_source(
            source_order,
            positions,
            traversal_atlas,
            selection.environment_world_id,
            geometry,
            role_mask,
            params,
        )
        edge_slots = np.flatnonzero(edge_mask[source])
        edge_slot = int(random.choice(edge_slots))
        destination = int(traversal_atlas.edge_destination[0, source, edge_slot])
        destination_rank = 1
        destination_method = "seeded_outgoing_traversal_edge_v1"
    elif require_initial_los:
        source, destination, source_rank, destination_rank = (
            _first_actor_legal_visible_region_pair(
                source_order,
                positions,
                node_mask,
                edge_mask,
                np.asarray(traversal_atlas.edge_destination[0]),
                inside_core,
                requested_separation,
                traversal_atlas,
                selection.environment_world_id,
                geometry,
                role_mask,
                params,
                random,
            )
        )
        destination_method = (
            "seeded_graph_reachable_actor_legal_visible_pair_ranked_batched_v4"
        )
    else:
        source, destination, source_rank, destination_rank = (
            _first_traversal_legal_region_pair(
                source_order,
                positions,
                node_mask,
                edge_mask,
                np.asarray(traversal_atlas.edge_destination[0]),
                inside_core,
                requested_separation,
                random,
            )
        )
        destination_method = "seeded_actor_profile_traversal_pair_ranked_v1"
    source_position = positions[source]
    target_position = positions[destination]
    actual_separation = float(
        np.linalg.norm(target_position[[0, 2]] - source_position[[0, 2]])
    )
    params = params._replace(
        agent_spawn=jnp.asarray(source_position, dtype=jnp.float32),
        target_offset=jnp.asarray(
            target_position - source_position,
            dtype=jnp.float32,
        ),
        floor_y=jnp.asarray(source_position[1], dtype=jnp.float32),
    )

    metadata = {
        "usage": REGION_ARTIFACT_USAGE_TRAINING,
        "region_library_semantic_sha256": (
            selection.working_set.library_semantic_sha256
        ),
        "traversal_library_semantic_sha256": (args.expected_traversal_sha256.lower()),
        "region_capture_evidence_bridge_sha256": (
            region_manifest["capture_contract"]["native_evidence_jar_sha256"]
        ),
        "traversal_evidence_bridge_sha256": (
            traversal_manifest["native_evidence_jar_sha256"]
        ),
        # These two name the FIRST resident world only. On a multi-world working
        # set they describe one row of `environment_world_id`, not the batch, so
        # a consumer that stamps them onto a per-environment record is claiming
        # the wrong world for every row that is not in slot 0. The tuples below
        # are the whole working set, indexed by `environment_world_id`; use them
        # whenever the thing being labelled belongs to one environment.
        "artifact_semantic_sha256": (selection.working_set.artifact_semantic_sha256[0]),
        "artifact_seed": selection.working_set.artifact_seed[0],
        "resident_artifact_semantic_sha256": [
            str(value) for value in selection.working_set.artifact_semantic_sha256
        ],
        "resident_artifact_seed": [
            int(value) for value in selection.working_set.artifact_seed
        ],
        "working_set_capacity": working_set_capacity,
        "environment_diversity": environment_diversity,
        "environment_world_id": np.asarray(
            selection.environment_world_id
        ).tolist(),
        "distinct_environment_worlds": int(
            np.unique(np.asarray(selection.environment_world_id)).size
        ),
        "environment_count": args.batch_size,
        "selection_key": args.region_selection_key,
        "assignment_key": args.region_assignment_key,
        "node_seed": args.region_node_seed,
        "source_selection_method": (
            "first_actor_legal_source_in_seeded_permutation_v1"
        ),
        "source_candidate_count": int(candidates.size),
        "source_selection_rank": source_rank,
        "source_rejections": source_rank - 1,
        "source_node": source,
        "destination_node": destination,
        "destination_selection_method": destination_method,
        "destination_selection_rank": destination_rank,
        "target_separation_range": requested_separation,
        "target_initial_horizontal_separation": actual_separation,
        "target_initial_line_of_sight_required": require_initial_los,
        "target_initial_line_of_sight_certified": (
            requested_separation is not None and require_initial_los
        ),
        "action_surface": _region_action_surface_metadata(
            getattr(args, "region_action_surface_config", None)
        ),
        "entity_only_explosion": _region_explosion_metadata(
            getattr(args, "region_action_surface_config", None)
        ),
        "region_surrogate_sky_light": _region_light_metadata(
            getattr(args, "region_native_light_config", None)
        )["surrogate_fallback"],
        "region_light": _region_light_metadata(
            getattr(args, "region_native_light_config", None)
        ),
    }
    if environment_agent_position is not None:
        # Carried in metadata rather than the return tuple: three callers
        # unpack that tuple positionally, and a silent arity change is exactly
        # the desync this fixture cannot afford.
        metadata["environment_agent_position"] = (
            environment_agent_position.tolist()
        )
        metadata["environment_target_position"] = (
            environment_target_position.tolist()
        )
        metadata["distinct_environment_spawns"] = int(
            np.unique(environment_agent_position, axis=0).shape[0]
        )
        metadata["distinct_environment_targets"] = int(
            np.unique(environment_target_position, axis=0).shape[0]
        )
        # Consumers check the opening separation against the actor sensor range,
        # so report the widest lane rather than row 0's: with per-row pairs the
        # lanes span the whole band, and a row-0-only figure lets a further lane
        # start unperceivable while the guard passes.
        separations = np.linalg.norm(
            environment_target_position[:, (0, 2)]
            - environment_agent_position[:, (0, 2)],
            axis=1,
        )
        metadata["target_initial_horizontal_separation"] = float(
            separations.max()
        )
        metadata["target_initial_horizontal_separation_minimum"] = float(
            separations.min()
        )
        metadata["target_initial_horizontal_separation_mean"] = float(
            separations.mean()
        )
    if action_surface_executor is not None:
        metadata["action_surface"].update(
            {
                "executor_contract_sha256": (
                    action_surface_executor.contract_sha256
                ),
                "excluded_stale_asset_references": list(
                    action_surface_executor.excluded_asset_references
                ),
            }
        )
    return (
        params,
        geometry,
        provider,
        token_provider,
        metadata,
        action_surface_runtime_initializer,
        world_runtime_provider,
        actor_world_runtime_provider,
        action_surface_provider,
        action_surface_executor,
        inventory_reset_provider,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _distributed_region_case_with_geometry(args, params, config):
    """Bind a filtered, artifact-backed Region reset distribution."""

    region_manifest = json.loads(
        args.region_manifest.read_text(encoding="utf-8")
    )
    traversal_manifest = json.loads(
        args.traversal_manifest.read_text(encoding="utf-8")
    )
    selection = region_environments_from_artifact_manifest(
        args.region_manifest,
        usage=REGION_ARTIFACT_USAGE_TRAINING,
        expected_library_semantic_sha256=args.expected_region_sha256,
        working_set_capacity=args.working_set_capacity,
        environment_count=args.batch_size,
        selection_key=args.region_selection_key,
        assignment_key=args.region_assignment_key,
        selection_seeds=getattr(args, "region_selection_seeds", None),
        traversal_manifest_path=args.traversal_manifest,
        expected_traversal_library_semantic_sha256=(
            args.expected_traversal_sha256
        ),
    )
    (
        geometry,
        provider,
        token_provider,
        _,
        action_surface_runtime_initializer,
        world_runtime_provider,
        actor_world_runtime_provider,
        action_surface_provider,
        action_surface_executor,
        inventory_reset_provider,
    ) = _region_world_bindings(
        selection,
        params,
        config,
        region_manifest_path=args.region_manifest,
        action_surface_config=getattr(
            args,
            "region_action_surface_config",
            None,
        ),
        native_light_config=getattr(
            args,
            "region_native_light_config",
            None,
        ),
    )

    report = json.loads(
        args.reset_pool_report.read_text(encoding="utf-8")
    )
    expected_fields = {
        "schema": "hytalerl_region_combat_reset_pool_v2",
        "usage": REGION_ARTIFACT_USAGE_TRAINING,
        "combat_filter_contract": region_reset_filter_combat_contract(params),
        "actor_evidence_contract": region_reset_actor_evidence_contract(
            token_capacity=DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG.token_capacity,
            maximum_distance=(
                DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG.maximum_distance
            ),
        ),
        "region_library_semantic_sha256": (
            args.expected_region_sha256.lower()
        ),
        "traversal_library_semantic_sha256": (
            args.expected_traversal_sha256.lower()
        ),
        "working_set_capacity": args.working_set_capacity,
        "selection_key": args.region_selection_key,
        "assignment_key": args.region_assignment_key,
        "pool_semantic_sha256": args.expected_reset_pool_sha256.upper(),
    }
    for name, expected in expected_fields.items():
        if report.get(name) != expected:
            raise ValueError(
                f"reset pool {name} mismatch: "
                f"{report.get(name)!r} != {expected!r}"
            )
    pool_path = args.reset_pool_report.parent / report["pool_file"]
    pool_file_sha256 = _file_sha256(pool_path)
    if pool_file_sha256 != args.expected_reset_pool_file_sha256.upper():
        raise ValueError(
            "reset pool file SHA-256 mismatch: "
            f"{pool_file_sha256} != "
            f"{args.expected_reset_pool_file_sha256.upper()}"
        )
    if report.get("pool_file_sha256") != pool_file_sha256:
        raise ValueError("reset pool report does not identify its NPZ bytes")

    pool = load_region_combat_reset_pool(pool_path)
    try:
        semantic_metadata = {
            field: report[field] for field in _RESET_POOL_SEMANTIC_FIELDS
        }
    except KeyError as error:
        raise ValueError(
            f"reset pool report lacks semantic field {error.args[0]!r}"
        ) from error
    computed_pool_sha256 = region_combat_reset_pool_sha256(
        pool,
        semantic_metadata,
    )
    if computed_pool_sha256 != report["pool_semantic_sha256"]:
        raise ValueError(
            "reset pool semantic SHA-256 does not match its metadata and arrays"
        )
    if pool.source_mask.shape[0] != args.working_set_capacity:
        raise ValueError("reset pool world axis does not match working set")
    report_worlds = report.get("worlds")
    if not isinstance(report_worlds, list) or len(report_worlds) != (
        args.working_set_capacity
    ):
        raise ValueError("reset pool report has the wrong world census")
    for world_id, world in enumerate(report_worlds):
        expected_world = {
            "world_id": world_id,
            "artifact_seed": int(
                selection.working_set.artifact_seed[world_id]
            ),
            "artifact_semantic_sha256": (
                selection.working_set.artifact_semantic_sha256[world_id]
            ),
        }
        for name, expected in expected_world.items():
            if world.get(name) != expected:
                raise ValueError(
                    f"reset pool world {world_id} {name} mismatch"
                )

    reset_provider = make_region_combat_reset_provider(
        pool,
        selection.environment_world_id,
        geometry=geometry,
    )
    metadata = {
        "usage": REGION_ARTIFACT_USAGE_TRAINING,
        "region_library_semantic_sha256": (
            selection.working_set.library_semantic_sha256
        ),
        "traversal_library_semantic_sha256": (
            args.expected_traversal_sha256.lower()
        ),
        "region_capture_evidence_bridge_sha256": (
            region_manifest["capture_contract"]["native_evidence_jar_sha256"]
        ),
        "traversal_evidence_bridge_sha256": (
            traversal_manifest["native_evidence_jar_sha256"]
        ),
        "artifact_semantic_sha256": list(
            selection.working_set.artifact_semantic_sha256
        ),
        "artifact_seed": list(selection.working_set.artifact_seed),
        "working_set_capacity": args.working_set_capacity,
        "environment_count": args.batch_size,
        "selection_key": args.region_selection_key,
        "assignment_key": args.region_assignment_key,
        "reset_distribution": {
            "schema": report["schema"],
            "pool_semantic_sha256": report["pool_semantic_sha256"],
            "pool_file_sha256": pool_file_sha256,
            "builder_sha256": report["builder_sha256"],
            "combat_filter_contract": report["combat_filter_contract"],
            "actor_evidence_contract": report["actor_evidence_contract"],
            "edge_selection_seed": report["edge_selection_seed"],
            "edge_selection_method": report["edge_selection_method"],
            "filter_seeds": report["filter_seeds"],
            "filter_steps": report["filter_steps"],
            "filter_semantics": report["filter_semantics"],
            "task_bias": report["task_bias"],
            "source_capacity": report["source_capacity"],
            "edge_capacity": report["edge_capacity"],
            "worlds": report_worlds,
            "core_admission": {
                "contract": region_combat_reset_core_admission_contract(),
                "contract_sha256": (
                    region_combat_reset_core_admission_contract_sha256()
                ),
            },
        },
        "action_surface": _region_action_surface_metadata(
            getattr(args, "region_action_surface_config", None)
        ),
        "entity_only_explosion": _region_explosion_metadata(
            getattr(args, "region_action_surface_config", None)
        ),
        "region_surrogate_sky_light": _region_light_metadata(
            getattr(args, "region_native_light_config", None)
        )["surrogate_fallback"],
        "region_light": _region_light_metadata(
            getattr(args, "region_native_light_config", None)
        ),
    }
    if action_surface_executor is not None:
        metadata["action_surface"].update(
            {
                "executor_contract_sha256": (
                    action_surface_executor.contract_sha256
                ),
                "excluded_stale_asset_references": list(
                    action_surface_executor.excluded_asset_references
                ),
            }
        )
    return (
        geometry,
        provider,
        token_provider,
        reset_provider,
        metadata,
        action_surface_runtime_initializer,
        world_runtime_provider,
        actor_world_runtime_provider,
        action_surface_provider,
        action_surface_executor,
        inventory_reset_provider,
    )


def _first_actor_legal_region_source(
    source_order,
    positions,
    traversal_atlas,
    environment_world_id,
    geometry,
    role_mask,
    params,
    *,
    line_of_sight_from=None,
):
    """Select the first actor-legal node with optional certified initial LOS."""

    batch = int(environment_world_id.shape[0])
    forward = jnp.broadcast_to(
        jnp.asarray((0.0, 0.0, -1.0), dtype=jnp.float32),
        (batch, 1, 3),
    )

    @jax.jit
    def actor_evidence(source_position):
        actor_position = jnp.broadcast_to(source_position, (batch, 1, 3))
        tokens = produce_actor_region_world_geometry_tokens(
            traversal_atlas,
            environment_world_id,
            geometry,
            actor_position,
            actor_position + params.agent_eye_offset[None, None, :],
            forward,
            role_opaque_mask=role_mask,
            geometry_provenance=(WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY),
            traversal_staging_capacity=(REGION_TRAVERSAL_POLICY_STAGING_CAPACITY),
            token_capacity=DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG.token_capacity,
            maximum_distance=(DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG.maximum_distance),
        )
        if line_of_sight_from is None:
            return tokens, jnp.ones((batch,), dtype=jnp.bool_)
        start = jnp.broadcast_to(
            jnp.asarray(line_of_sight_from, dtype=jnp.float32)
            + params.agent_eye_offset,
            (batch, 3),
        )
        end = jnp.broadcast_to(
            source_position + params.target_eye_offset,
            (batch, 3),
        )
        line_of_sight = geometry_perception_line_of_sight_result(
            geometry,
            start,
            end,
            role_opaque_mask=None,
        )
        certified = (
            line_of_sight.visible
            & ~line_of_sight.geometry_exhausted
            & ~line_of_sight.capacity_exceeded
            & ~line_of_sight.invalid
        )
        return tokens, certified

    for rank, source in enumerate(source_order, start=1):
        tokens, certified_los = actor_evidence(
            jnp.asarray(positions[int(source)], dtype=jnp.float32)
        )
        available = np.asarray(jax.device_get(tokens.available))
        token_mask = np.asarray(jax.device_get(tokens.token_mask))
        if (
            np.all(available)
            and np.all(np.any(token_mask, axis=2))
            and np.all(np.asarray(jax.device_get(certified_los)))
        ):
            return int(source), rank
    evidence = " with certified line of sight" if line_of_sight_from is not None else ""
    raise RuntimeError(
        f"selected Region has no actor-legal core traversal source{evidence}"
    )


if __name__ == "__main__":
    main()
