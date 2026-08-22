"""Pinned minigame recipes over exact WorldGen V2 Region artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.worldgen.block_actions import BLOCK_SEMANTIC_KEY_WORDS
from hytalegym.worldgen.region import (
    CHUNK_SIZE,
    CORE_BLOCKS_PER_AXIS,
    MIN_Y,
    WORLD_HEIGHT,
)

from ..bundles.bundle import (
    CAPABILITY_BIOME_IDENTITY,
    CAPABILITY_DYNAMIC_ENTITIES,
    CAPABILITY_EXACT_GEOMETRY,
    CAPABILITY_NATIVE_LIGHTING,
    CAPABILITY_NATIVE_TRAVERSAL,
    CAPABILITY_STABLE_BLOCKS,
    CAPABILITY_STABLE_FLUIDS,
    CAPABILITY_STATIC_SPAWN,
    CAPABILITY_STRUCTURE_INSTANCES,
    V2JaxArtifact,
    V2JaxBundle,
)
from ..bundles.jax_loader import V2JaxWorldSet, materialize_jax_worlds
from ..structures.structure_pack import V2StructurePack, V2StructureSnapshot
from ..bundles.traversal_pack import V2TraversalPack


ENVIRONMENT_RECIPE_SCHEMA = "hytalerl_worldgen_v2_environment_recipe_v1"
ENVIRONMENT_RECIPE_VERSION = 1

OBJECTIVE_SANDBOX = 0
OBJECTIVE_REACH = 1
OBJECTIVE_CHECKPOINT = 2
OBJECTIVE_ELIMINATE = 3
OBJECTIVE_SURVIVE = 4

_OBJECTIVE_CODE = {
    "sandbox": OBJECTIVE_SANDBOX,
    "reach": OBJECTIVE_REACH,
    "checkpoint": OBJECTIVE_CHECKPOINT,
    "eliminate": OBJECTIVE_ELIMINATE,
    "survive": OBJECTIVE_SURVIVE,
}
_CAPABILITIES = frozenset(
    {
        CAPABILITY_EXACT_GEOMETRY,
        CAPABILITY_STABLE_BLOCKS,
        CAPABILITY_STABLE_FLUIDS,
        CAPABILITY_STATIC_SPAWN,
        CAPABILITY_NATIVE_TRAVERSAL,
        CAPABILITY_STRUCTURE_INSTANCES,
        CAPABILITY_BIOME_IDENTITY,
        CAPABILITY_DYNAMIC_ENTITIES,
        CAPABILITY_NATIVE_LIGHTING,
    }
)
_SIDECAR_CAPABILITIES = frozenset(
    {CAPABILITY_NATIVE_TRAVERSAL, CAPABILITY_STRUCTURE_INSTANCES}
)
_RECIPE_FIELDS = frozenset(
    {
        "schema",
        "version",
        "recipe_id",
        "source_bundle_semantic_sha256",
        "worlds",
        "required_capabilities",
        "bounds",
        "teams",
        "objective",
        "quality_gates",
        "recipe_semantic_sha256",
    }
)
_WORLD_FIELDS = frozenset(
    {"structure", "seed", "region_semantic_sha256"}
)
_BOUNDS_FIELDS = frozenset({"coordinate_mode", "minimum", "maximum"})
_TEAM_FIELDS = frozenset({"team_id", "spawn"})
_NATIVE_SPAWN_FIELDS = frozenset({"kind", "offset"})
_FIXED_POSITION_FIELDS = frozenset({"kind", "coordinate_mode", "position"})
_STRUCTURE_POSITION_FIELDS = frozenset(
    {"kind", "structure_asset_id", "occurrence", "local_offset"}
)
_NEAREST_STRUCTURE_POSITION_FIELDS = frozenset(
    {"kind", "structure_asset_id", "local_offset"}
)
_QUALITY_FIELDS = frozenset(
    {
        "minimum_terrain_relief_blocks",
        "minimum_unique_non_air_material_ids",
        "minimum_full_fluid_cells",
        "required_material_asset_ids",
        "required_fluid_asset_ids",
        "required_structure_asset_ids",
    }
)
_COORDINATE_MODES = frozenset({"region_core", "world"})
_STRUCTURE_POSITION_KINDS = frozenset(
    {"structure_anchor", "nearest_structure_anchor"}
)
_POSITION_KINDS = frozenset(
    {"native_spawn", "fixed"} | _STRUCTURE_POSITION_KINDS
)
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")
_KEY_FORMAT = f">{BLOCK_SEMANTIC_KEY_WORDS}I"


@dataclass(frozen=True)
class V2EnvironmentBounds:
    coordinate_mode: str
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]


@dataclass(frozen=True)
class V2PositionRule:
    kind: str
    coordinate_mode: str | None = None
    position: tuple[float, float, float] | None = None
    offset: tuple[float, float, float] | None = None
    structure_asset_id: str | None = None
    occurrence: int | None = None
    local_offset: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class V2TeamDefinition:
    team_id: str
    spawn: V2PositionRule


@dataclass(frozen=True)
class V2ObjectiveDefinition:
    kind: str
    points: tuple[V2PositionRule, ...] = ()
    target_team_id: str | None = None
    duration_steps: int = 0
    radius: float = 1.25


@dataclass(frozen=True)
class V2QualityGates:
    minimum_terrain_relief_blocks: int
    minimum_unique_non_air_material_ids: int
    minimum_full_fluid_cells: int
    required_material_asset_ids: tuple[str, ...]
    required_fluid_asset_ids: tuple[str, ...]
    required_structure_asset_ids: tuple[str, ...]


@dataclass(frozen=True)
class V2EnvironmentRecipe:
    """One immutable environment design over an explicit artifact allowlist."""

    path: Path
    recipe_id: str
    source_bundle_semantic_sha256: str
    artifacts: tuple[V2JaxArtifact, ...]
    required_capabilities: frozenset[str]
    bounds: V2EnvironmentBounds
    teams: tuple[V2TeamDefinition, ...]
    objective: V2ObjectiveDefinition
    quality_gates: V2QualityGates
    semantic_sha256: str
    manifest: Mapping[str, Any]

    @classmethod
    def load(
        cls,
        path: str | Path,
        bundle: V2JaxBundle,
    ) -> "V2EnvironmentRecipe":
        if not isinstance(bundle, V2JaxBundle):
            raise TypeError("bundle must be a V2JaxBundle")
        source = Path(path).resolve()
        value = _json_object(source)
        _exact_fields(value, _RECIPE_FIELDS, "environment recipe")
        if value["schema"] != ENVIRONMENT_RECIPE_SCHEMA:
            raise ValueError("unsupported V2 environment recipe schema")
        if _exact_int(value["version"], "recipe version") != (
            ENVIRONMENT_RECIPE_VERSION
        ):
            raise ValueError("unsupported V2 environment recipe version")
        recipe_id = _safe_id(value["recipe_id"], "recipe_id")
        bundle_semantic = _sha256(
            value["source_bundle_semantic_sha256"],
            "source bundle semantic",
        )
        if bundle_semantic != bundle.semantic_sha256:
            raise ValueError("environment recipe names another V2 bundle")
        semantic = _sha256(
            value["recipe_semantic_sha256"], "recipe semantic"
        )
        if semantic != _semantic_digest(value):
            raise ValueError("V2 environment recipe semantic SHA-256 mismatch")
        artifacts = _resolve_worlds(value["worlds"], bundle)
        capabilities = _capability_set(value["required_capabilities"])
        bounds = _bounds(value["bounds"])
        teams = _teams(value["teams"])
        objective = _objective(value["objective"], teams)
        quality = _quality_gates(value["quality_gates"])
        derived = _derived_capabilities(bounds, teams, objective, quality)
        missing_declarations = derived - capabilities
        if missing_declarations:
            raise ValueError(
                "environment recipe omits derived capabilities: "
                f"{sorted(missing_declarations)}"
            )
        _validate_base_capabilities(bundle, capabilities)
        _validate_quality(artifacts, quality)
        for artifact in artifacts:
            resolved_bounds = _resolve_bounds(bounds, artifact)
            for team in teams:
                if team.spawn.kind not in _STRUCTURE_POSITION_KINDS:
                    _require_inside(
                        _resolve_position(
                            team.spawn,
                            artifact,
                            None,
                            resolved_bounds=resolved_bounds,
                        ),
                        resolved_bounds,
                        f"team spawn {team.team_id}",
                    )
            for index, point in enumerate(objective.points):
                if point.kind not in _STRUCTURE_POSITION_KINDS:
                    _require_inside(
                        _resolve_position(
                            point,
                            artifact,
                            None,
                            resolved_bounds=resolved_bounds,
                        ),
                        resolved_bounds,
                        f"objective point {index}",
                    )
        return cls(
            path=source,
            recipe_id=recipe_id,
            source_bundle_semantic_sha256=bundle_semantic,
            artifacts=artifacts,
            required_capabilities=capabilities,
            bounds=bounds,
            teams=teams,
            objective=objective,
            quality_gates=quality,
            semantic_sha256=semantic,
            manifest=_json_copy(value),
        )


class V2EnvironmentRuntime(NamedTuple):
    """Fixed-shape environment design arrays indexed by JAX environments."""

    bounds_min: jax.Array
    bounds_max: jax.Array
    team_mask: jax.Array
    team_key: jax.Array
    team_spawn_position: jax.Array
    objective_kind: jax.Array
    objective_point_mask: jax.Array
    objective_position: jax.Array
    objective_radius: jax.Array
    objective_target_team_key: jax.Array
    objective_duration_steps: jax.Array


@dataclass(frozen=True)
class V2JaxEnvironmentBatch:
    """Exact V2 worlds plus one deliberate minigame recipe."""

    worlds: V2JaxWorldSet
    runtime: V2EnvironmentRuntime
    recipe_id: str
    team_ids: tuple[str, ...]
    source_recipe_semantic_sha256: str


def create_environment_recipe(
    path: str | Path,
    bundle: V2JaxBundle,
    artifacts: Sequence[V2JaxArtifact],
    *,
    recipe_id: str,
    required_capabilities: Sequence[str],
    bounds: Mapping[str, Any],
    teams: Sequence[Mapping[str, Any]],
    objective: Mapping[str, Any],
    quality_gates: Mapping[str, Any],
) -> V2EnvironmentRecipe:
    """Normalize, hash, publish, and reload one environment recipe."""

    if not isinstance(bundle, V2JaxBundle):
        raise TypeError("bundle must be a V2JaxBundle")
    selected = tuple(artifacts)
    if not selected:
        raise ValueError("environment recipe artifacts cannot be empty")
    known = {
        (row.structure, row.seed, row.region_semantic_sha256): row
        for row in bundle.artifacts
    }
    if any(
        known.get((row.structure, row.seed, row.region_semantic_sha256)) != row
        for row in selected
    ):
        raise ValueError("environment recipe artifact does not belong to bundle")
    parsed_bounds = _bounds(bounds)
    parsed_teams = _teams(list(teams))
    parsed_objective = _objective(objective, parsed_teams)
    parsed_quality = _quality_gates(quality_gates)
    capabilities = _capability_set(list(required_capabilities))
    value: dict[str, Any] = {
        "schema": ENVIRONMENT_RECIPE_SCHEMA,
        "version": ENVIRONMENT_RECIPE_VERSION,
        "recipe_id": _safe_id(recipe_id, "recipe_id"),
        "source_bundle_semantic_sha256": bundle.semantic_sha256,
        "worlds": [
            {
                "structure": row.structure,
                "seed": row.seed,
                "region_semantic_sha256": row.region_semantic_sha256,
            }
            for row in selected
        ],
        "required_capabilities": sorted(capabilities),
        "bounds": _bounds_value(parsed_bounds),
        "teams": [_team_value(row) for row in parsed_teams],
        "objective": _objective_value(parsed_objective),
        "quality_gates": _quality_value(parsed_quality),
        "recipe_semantic_sha256": "",
    }
    value["recipe_semantic_sha256"] = _semantic_digest(value)
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination, value)
    return V2EnvironmentRecipe.load(destination, bundle)


def materialize_environment_recipe(
    recipe: V2EnvironmentRecipe,
    bundle: V2JaxBundle,
    *,
    environment_count: int,
    assignment_key: int = 0,
    region_capacity: int | None = None,
    cell_palette_capacity: int | None = None,
    shape_palette_capacity: int | None = None,
    block_palette_capacity: int | None = None,
    fluid_palette_capacity: int | None = None,
    traversal_pack: V2TraversalPack | None = None,
    traversal_node_capacity: int | None = None,
    traversal_column_node_capacity: int | None = None,
    structure_pack: V2StructurePack | None = None,
    structure_instance_capacity: int | None = None,
) -> V2JaxEnvironmentBatch:
    """Load exact artifacts and resolve recipe rules once, before JIT use."""

    if not isinstance(recipe, V2EnvironmentRecipe):
        raise TypeError("recipe must be a V2EnvironmentRecipe")
    if not isinstance(bundle, V2JaxBundle):
        raise TypeError("bundle must be a V2JaxBundle")
    if recipe.source_bundle_semantic_sha256 != bundle.semantic_sha256:
        raise ValueError("environment recipe and V2 bundle differ")
    worlds = materialize_jax_worlds(
        bundle,
        recipe.artifacts,
        environment_count=environment_count,
        assignment_key=assignment_key,
        region_capacity=region_capacity,
        cell_palette_capacity=cell_palette_capacity,
        shape_palette_capacity=shape_palette_capacity,
        block_palette_capacity=block_palette_capacity,
        fluid_palette_capacity=fluid_palette_capacity,
        traversal_pack=traversal_pack,
        traversal_node_capacity=traversal_node_capacity,
        traversal_column_node_capacity=traversal_column_node_capacity,
        structure_pack=structure_pack,
        structure_instance_capacity=structure_instance_capacity,
        required_capabilities=tuple(recipe.required_capabilities),
    )
    structure_snapshots = None
    if structure_pack is not None:
        structure_snapshots = structure_pack.load_for_artifacts(recipe.artifacts)
    runtime = _compile_runtime(recipe, worlds, structure_snapshots)
    return V2JaxEnvironmentBatch(
        worlds=worlds,
        runtime=runtime,
        recipe_id=recipe.recipe_id,
        team_ids=tuple(row.team_id for row in recipe.teams),
        source_recipe_semantic_sha256=recipe.semantic_sha256,
    )


def _compile_runtime(
    recipe: V2EnvironmentRecipe,
    worlds: V2JaxWorldSet,
    structure_snapshots: Sequence[V2StructureSnapshot] | None,
) -> V2EnvironmentRuntime:
    world_count = len(recipe.artifacts)
    team_count = len(recipe.teams)
    point_count = max(len(recipe.objective.points), 1)
    bounds_min = np.zeros((world_count, 3), dtype=np.float32)
    bounds_max = np.zeros((world_count, 3), dtype=np.float32)
    team_mask = np.ones((world_count, team_count), dtype=np.bool_)
    team_key = np.broadcast_to(
        np.asarray([_asset_key(row.team_id) for row in recipe.teams], dtype=np.uint32),
        (world_count, team_count, BLOCK_SEMANTIC_KEY_WORDS),
    ).copy()
    team_spawn = np.zeros((world_count, team_count, 3), dtype=np.float32)
    objective_point_mask = np.zeros((world_count, point_count), dtype=np.bool_)
    objective_position = np.zeros((world_count, point_count, 3), dtype=np.float32)
    objective_kind = np.full(
        world_count,
        _OBJECTIVE_CODE[recipe.objective.kind],
        dtype=np.uint8,
    )
    target_key = np.zeros(
        (world_count, BLOCK_SEMANTIC_KEY_WORDS), dtype=np.uint32
    )
    if recipe.objective.target_team_id is not None:
        target_key[:] = _asset_key(recipe.objective.target_team_id)
    duration = np.full(
        world_count,
        recipe.objective.duration_steps,
        dtype=np.int32,
    )
    radius = np.full(
        world_count,
        recipe.objective.radius,
        dtype=np.float32,
    )
    for world_index, artifact in enumerate(recipe.artifacts):
        snapshot = (
            None
            if structure_snapshots is None
            else structure_snapshots[world_index]
        )
        resolved_bounds = _resolve_bounds(recipe.bounds, artifact)
        bounds_min[world_index] = resolved_bounds[0]
        bounds_max[world_index] = resolved_bounds[1]
        _validate_required_structures(recipe.quality_gates, snapshot, artifact)
        for team_index, team in enumerate(recipe.teams):
            position = _resolve_position(
                team.spawn,
                artifact,
                snapshot,
                resolved_bounds=resolved_bounds,
            )
            _require_inside(position, resolved_bounds, f"team spawn {team.team_id}")
            team_spawn[world_index, team_index] = position
        for point_index, rule in enumerate(recipe.objective.points):
            position = _resolve_position(
                rule,
                artifact,
                snapshot,
                resolved_bounds=resolved_bounds,
            )
            _require_inside(
                position,
                resolved_bounds,
                f"objective point {point_index}",
            )
            objective_point_mask[world_index, point_index] = True
            objective_position[world_index, point_index] = position
    environment_world_id = worlds.environment_world_id
    return V2EnvironmentRuntime(
        bounds_min=jnp.asarray(bounds_min)[environment_world_id],
        bounds_max=jnp.asarray(bounds_max)[environment_world_id],
        team_mask=jnp.asarray(team_mask)[environment_world_id],
        team_key=jnp.asarray(team_key)[environment_world_id],
        team_spawn_position=jnp.asarray(team_spawn)[environment_world_id],
        objective_kind=jnp.asarray(objective_kind)[environment_world_id],
        objective_point_mask=jnp.asarray(objective_point_mask)[
            environment_world_id
        ],
        objective_position=jnp.asarray(objective_position)[environment_world_id],
        objective_radius=jnp.asarray(radius)[environment_world_id],
        objective_target_team_key=jnp.asarray(target_key)[environment_world_id],
        objective_duration_steps=jnp.asarray(duration)[environment_world_id],
    )


def _resolve_worlds(value: object, bundle: V2JaxBundle) -> tuple[V2JaxArtifact, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("environment recipe worlds must be a non-empty list")
    by_identity = {
        (row.structure, row.seed, row.region_semantic_sha256): row
        for row in bundle.artifacts
    }
    result = []
    for raw in value:
        row = _mapping(raw, "recipe world")
        _exact_fields(row, _WORLD_FIELDS, "recipe world")
        identity = (
            _nonempty(row["structure"], "world structure"),
            _nonnegative_int(row["seed"], "world seed"),
            _sha256(row["region_semantic_sha256"], "world Region semantic"),
        )
        artifact = by_identity.get(identity)
        if artifact is None:
            raise ValueError(f"environment recipe world unavailable: {identity[:2]}")
        result.append(artifact)
    identities = [
        (row.structure, row.seed, row.region_semantic_sha256) for row in result
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("environment recipe worlds must be unique")
    return tuple(result)


def _capability_set(value: object) -> frozenset[str]:
    if not isinstance(value, list) or any(not isinstance(row, str) for row in value):
        raise TypeError("required_capabilities must be a list of strings")
    if value != sorted(set(value)):
        raise ValueError("required_capabilities must be unique and sorted")
    result = frozenset(value)
    unknown = result - _CAPABILITIES
    if unknown:
        raise ValueError(f"unknown V2 JAX capabilities: {sorted(unknown)}")
    if CAPABILITY_EXACT_GEOMETRY not in result:
        raise ValueError("environment recipes must require exact Region geometry")
    return result


def _bounds(value: object) -> V2EnvironmentBounds:
    row = _mapping(value, "environment bounds")
    _exact_fields(row, _BOUNDS_FIELDS, "environment bounds")
    mode = _coordinate_mode(row["coordinate_mode"])
    minimum = _float3(row["minimum"], "bounds minimum")
    maximum = _float3(row["maximum"], "bounds maximum")
    if any(low >= high for low, high in zip(minimum, maximum, strict=True)):
        raise ValueError("environment bounds must be positive")
    if mode == "region_core" and (
        minimum[0] < 0.0
        or minimum[2] < 0.0
        or maximum[0] > CORE_BLOCKS_PER_AXIS
        or maximum[2] > CORE_BLOCKS_PER_AXIS
        or minimum[1] < MIN_Y
        or maximum[1] > MIN_Y + WORLD_HEIGHT
    ):
        raise ValueError("region_core bounds exceed the exact Region core")
    return V2EnvironmentBounds(mode, minimum, maximum)


def _teams(value: object) -> tuple[V2TeamDefinition, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("environment recipe teams must be a non-empty list")
    result = []
    for raw in value:
        row = _mapping(raw, "team definition")
        _exact_fields(row, _TEAM_FIELDS, "team definition")
        result.append(
            V2TeamDefinition(
                team_id=_safe_id(row["team_id"], "team_id"),
                spawn=_position_rule(row["spawn"]),
            )
        )
    if len({row.team_id for row in result}) != len(result):
        raise ValueError("environment recipe team IDs must be unique")
    return tuple(result)


def _position_rule(value: object) -> V2PositionRule:
    row = _mapping(value, "position rule")
    kind = _nonempty(row.get("kind"), "position rule kind")
    if kind not in _POSITION_KINDS:
        raise ValueError("unsupported V2 environment position rule")
    if kind == "native_spawn":
        _exact_fields(row, _NATIVE_SPAWN_FIELDS, "native spawn rule")
        return V2PositionRule(
            kind=kind,
            offset=_float3(row["offset"], "native spawn offset"),
        )
    if kind == "fixed":
        _exact_fields(row, _FIXED_POSITION_FIELDS, "fixed position rule")
        return V2PositionRule(
            kind=kind,
            coordinate_mode=_coordinate_mode(row["coordinate_mode"]),
            position=_float3(row["position"], "fixed position"),
        )
    expected = (
        _STRUCTURE_POSITION_FIELDS
        if kind == "structure_anchor"
        else _NEAREST_STRUCTURE_POSITION_FIELDS
    )
    _exact_fields(row, expected, "structure position rule")
    return V2PositionRule(
        kind=kind,
        structure_asset_id=_asset_id(
            row["structure_asset_id"], "structure asset ID"
        ),
        occurrence=(
            _nonnegative_int(row["occurrence"], "structure occurrence")
            if kind == "structure_anchor"
            else None
        ),
        local_offset=_float3(row["local_offset"], "structure local offset"),
    )


def _objective(
    value: object,
    teams: Sequence[V2TeamDefinition],
) -> V2ObjectiveDefinition:
    row = _mapping(value, "objective")
    kind = _nonempty(row.get("kind"), "objective kind")
    if kind not in _OBJECTIVE_CODE:
        raise ValueError("unsupported V2 environment objective")
    if kind == "sandbox":
        _exact_fields(row, frozenset({"kind"}), "sandbox objective")
        return V2ObjectiveDefinition(kind=kind)
    if kind == "reach":
        _optional_radius_fields(row, frozenset({"kind", "point"}), "reach")
        return V2ObjectiveDefinition(
            kind=kind,
            points=(_position_rule(row["point"]),),
            radius=_objective_radius(row),
        )
    if kind == "checkpoint":
        _optional_radius_fields(
            row, frozenset({"kind", "points"}), "checkpoint"
        )
        points = row["points"]
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError("checkpoint objective requires at least two points")
        return V2ObjectiveDefinition(
            kind=kind,
            points=tuple(_position_rule(point) for point in points),
            radius=_objective_radius(row),
        )
    if kind == "eliminate":
        _exact_fields(
            row, frozenset({"kind", "target_team_id"}), "eliminate objective"
        )
        target = _safe_id(row["target_team_id"], "target_team_id")
        if target not in {team.team_id for team in teams}:
            raise ValueError("eliminate objective names an unknown team")
        return V2ObjectiveDefinition(kind=kind, target_team_id=target)
    _exact_fields(
        row, frozenset({"kind", "duration_steps"}), "survive objective"
    )
    return V2ObjectiveDefinition(
        kind=kind,
        duration_steps=_positive_int(row["duration_steps"], "duration_steps"),
    )


def _quality_gates(value: object) -> V2QualityGates:
    row = _mapping(value, "quality gates")
    _exact_fields(row, _QUALITY_FIELDS, "quality gates")
    return V2QualityGates(
        minimum_terrain_relief_blocks=_nonnegative_int(
            row["minimum_terrain_relief_blocks"], "minimum terrain relief"
        ),
        minimum_unique_non_air_material_ids=_nonnegative_int(
            row["minimum_unique_non_air_material_ids"],
            "minimum unique non-air materials",
        ),
        minimum_full_fluid_cells=_nonnegative_int(
            row["minimum_full_fluid_cells"], "minimum full fluid cells"
        ),
        required_material_asset_ids=_asset_id_list(
            row["required_material_asset_ids"], "required material asset IDs"
        ),
        required_fluid_asset_ids=_asset_id_list(
            row["required_fluid_asset_ids"], "required fluid asset IDs"
        ),
        required_structure_asset_ids=_asset_id_list(
            row["required_structure_asset_ids"], "required structure asset IDs"
        ),
    )


def _derived_capabilities(
    bounds: V2EnvironmentBounds,
    teams: Sequence[V2TeamDefinition],
    objective: V2ObjectiveDefinition,
    quality: V2QualityGates,
) -> frozenset[str]:
    del bounds
    result = {CAPABILITY_EXACT_GEOMETRY}
    rules = [team.spawn for team in teams] + list(objective.points)
    if any(rule.kind == "native_spawn" for rule in rules):
        result.add(CAPABILITY_STATIC_SPAWN)
    if any(rule.kind in _STRUCTURE_POSITION_KINDS for rule in rules) or (
        quality.required_structure_asset_ids
    ):
        result.add(CAPABILITY_STRUCTURE_INSTANCES)
    if quality.minimum_unique_non_air_material_ids or (
        quality.required_material_asset_ids
    ):
        result.add(CAPABILITY_STABLE_BLOCKS)
    if quality.minimum_full_fluid_cells or quality.required_fluid_asset_ids:
        result.add(CAPABILITY_STABLE_FLUIDS)
    return frozenset(result)


def _validate_base_capabilities(
    bundle: V2JaxBundle,
    capabilities: frozenset[str],
) -> None:
    bundle.require_capabilities(tuple(capabilities - _SIDECAR_CAPABILITIES))


def _validate_quality(
    artifacts: Sequence[V2JaxArtifact],
    quality: V2QualityGates,
) -> None:
    for artifact in artifacts:
        features = artifact.features
        label = f"{artifact.structure}/{artifact.seed}"
        for key, minimum in (
            ("terrain_relief_blocks", quality.minimum_terrain_relief_blocks),
            (
                "unique_non_air_material_ids",
                quality.minimum_unique_non_air_material_ids,
            ),
            ("full_fluid_cells", quality.minimum_full_fluid_cells),
        ):
            value = features.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"V2 feature {key} is not numeric: {label}")
            if value < minimum:
                raise ValueError(
                    f"V2 environment quality gate {key} failed: {label}"
                )
        _require_assets(
            quality.required_material_asset_ids,
            features.get("non_air_material_asset_ids"),
            "material",
            label,
        )
        _require_assets(
            quality.required_fluid_asset_ids,
            features.get("fluid_asset_ids"),
            "fluid",
            label,
        )


def _validate_required_structures(
    quality: V2QualityGates,
    snapshot: V2StructureSnapshot | None,
    artifact: V2JaxArtifact,
) -> None:
    required = set(quality.required_structure_asset_ids)
    if not required:
        return
    if snapshot is None:
        raise ValueError("required structures need a marker-backed structure pack")
    available = {row.structure_asset_id for row in snapshot.instances}
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            "V2 environment required structures unavailable for "
            f"{artifact.structure}/{artifact.seed}: {missing}"
        )


def _resolve_bounds(
    bounds: V2EnvironmentBounds,
    artifact: V2JaxArtifact,
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.asarray(bounds.minimum, dtype=np.float64)
    maximum = np.asarray(bounds.maximum, dtype=np.float64)
    origin = np.asarray(
        [
            artifact.core_min_chunk_xz[0] * CHUNK_SIZE,
            0.0,
            artifact.core_min_chunk_xz[1] * CHUNK_SIZE,
        ],
        dtype=np.float64,
    )
    if bounds.coordinate_mode == "region_core":
        minimum = minimum + origin
        maximum = maximum + origin
    core_min = origin + np.asarray([0.0, MIN_Y, 0.0])
    core_max = origin + np.asarray(
        [CORE_BLOCKS_PER_AXIS, MIN_Y + WORLD_HEIGHT, CORE_BLOCKS_PER_AXIS]
    )
    if np.any(minimum < core_min) or np.any(maximum > core_max):
        raise ValueError("environment bounds exceed an exact Region core")
    return minimum, maximum


def _resolve_position(
    rule: V2PositionRule,
    artifact: V2JaxArtifact,
    snapshot: V2StructureSnapshot | None,
    *,
    resolved_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    if rule.kind == "native_spawn":
        return np.asarray(artifact.spawn_position) + np.asarray(rule.offset)
    if rule.kind == "fixed":
        position = np.asarray(rule.position, dtype=np.float64)
        if rule.coordinate_mode == "region_core":
            position = position + np.asarray(
                [
                    artifact.core_min_chunk_xz[0] * CHUNK_SIZE,
                    0.0,
                    artifact.core_min_chunk_xz[1] * CHUNK_SIZE,
                ]
            )
        return position
    if snapshot is None:
        raise ValueError("structure-anchor rule requires a structure pack")
    matches = sorted(
        (
            row
            for row in snapshot.instances
            if row.structure_asset_id == rule.structure_asset_id
        ),
        key=_structure_geometry_key,
    )
    local = tuple(float(value) for value in rule.local_offset)

    def position(instance: Any) -> np.ndarray:
        rotated = _rotate_float_xz(local, instance.quarter_turns)
        return np.asarray(instance.anchor_world, dtype=np.float64) + np.asarray(
            rotated
        )

    if rule.kind == "structure_anchor":
        occurrence = int(rule.occurrence)
        if occurrence >= len(matches):
            raise ValueError(
                "structure-anchor occurrence unavailable: "
                f"{rule.structure_asset_id}/{occurrence}"
            )
        return position(matches[occurrence])

    spawn = np.asarray(artifact.spawn_position, dtype=np.float64)
    candidates = []
    for instance in matches:
        candidate = position(instance)
        inside = resolved_bounds is None or (
            np.all(candidate >= resolved_bounds[0])
            and np.all(candidate < resolved_bounds[1])
        )
        if (
            instance.capture_coverage == "complete"
            and instance.geometry_present
            and inside
        ):
            candidates.append(
                (
                    float(np.sum((candidate - spawn) ** 2)),
                    _structure_geometry_key(instance),
                    candidate,
                )
            )
    if not candidates:
        raise ValueError(
            "nearest structure anchor unavailable inside environment bounds: "
            f"{rule.structure_asset_id}"
        )
    return min(candidates, key=lambda row: (row[0], row[1]))[2]


def _structure_geometry_key(instance: Any) -> tuple[Any, ...]:
    """Order generated instances without native run-local identifiers.

    Hytale's prefab instance number is useful for grouping markers within one
    capture, but it is reassigned when the same seed is generated in a new
    server cycle.  Recipe resolution therefore orders by the captured authored
    geometry and metadata.  Identical keys resolve to identical positions, so
    their residual input order cannot affect a recipe objective.
    """

    return (
        tuple(instance.anchor_world),
        int(instance.quarter_turns),
        tuple(instance.bounds_min),
        tuple(instance.bounds_max),
        str(instance.structure_asset_id),
        str(instance.marker_asset_id),
        str(instance.kind),
        str(instance.capture_coverage),
        bool(instance.geometry_present),
    )


def _require_inside(
    position: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    label: str,
) -> None:
    point = np.asarray(position, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{label} must resolve to one finite XYZ position")
    if np.any(point < bounds[0]) or np.any(point >= bounds[1]):
        raise ValueError(f"{label} falls outside environment bounds")


def _bounds_value(value: V2EnvironmentBounds) -> dict[str, Any]:
    return {
        "coordinate_mode": value.coordinate_mode,
        "minimum": list(value.minimum),
        "maximum": list(value.maximum),
    }


def _team_value(value: V2TeamDefinition) -> dict[str, Any]:
    return {"team_id": value.team_id, "spawn": _position_rule_value(value.spawn)}


def _position_rule_value(value: V2PositionRule) -> dict[str, Any]:
    if value.kind == "native_spawn":
        return {"kind": value.kind, "offset": list(value.offset)}
    if value.kind == "fixed":
        return {
            "kind": value.kind,
            "coordinate_mode": value.coordinate_mode,
            "position": list(value.position),
        }
    result = {
        "kind": value.kind,
        "structure_asset_id": value.structure_asset_id,
        "local_offset": list(value.local_offset),
    }
    if value.kind == "structure_anchor":
        result["occurrence"] = value.occurrence
    return result


def _objective_value(value: V2ObjectiveDefinition) -> dict[str, Any]:
    if value.kind == "sandbox":
        return {"kind": value.kind}
    if value.kind == "reach":
        return {
            "kind": value.kind,
            "point": _position_rule_value(value.points[0]),
            "radius": value.radius,
        }
    if value.kind == "checkpoint":
        return {
            "kind": value.kind,
            "points": [_position_rule_value(row) for row in value.points],
            "radius": value.radius,
        }
    if value.kind == "eliminate":
        return {"kind": value.kind, "target_team_id": value.target_team_id}
    return {"kind": value.kind, "duration_steps": value.duration_steps}


def _quality_value(value: V2QualityGates) -> dict[str, Any]:
    return {
        "minimum_terrain_relief_blocks": value.minimum_terrain_relief_blocks,
        "minimum_unique_non_air_material_ids": (
            value.minimum_unique_non_air_material_ids
        ),
        "minimum_full_fluid_cells": value.minimum_full_fluid_cells,
        "required_material_asset_ids": list(value.required_material_asset_ids),
        "required_fluid_asset_ids": list(value.required_fluid_asset_ids),
        "required_structure_asset_ids": list(value.required_structure_asset_ids),
    }


def _semantic_digest(value: Mapping[str, Any]) -> str:
    stable = dict(value)
    stable["recipe_semantic_sha256"] = ""
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _asset_key(value: str) -> tuple[int, ...]:
    return struct.unpack(
        _KEY_FORMAT,
        hashlib.sha256(value.encode("utf-8")).digest(),
    )


def _rotate_float_xz(
    position: tuple[float, float, float],
    turns: int,
) -> tuple[float, float, float]:
    x, y, z = position
    if turns == 0:
        return x, y, z
    if turns == 1:
        return -z, y, x
    if turns == 2:
        return -x, y, -z
    return z, y, -x


def _require_assets(
    required: Sequence[str],
    available: object,
    kind: str,
    label: str,
) -> None:
    if not isinstance(available, list) or any(
        not isinstance(value, str) for value in available
    ):
        raise TypeError(f"V2 {kind} asset feature is invalid: {label}")
    missing = sorted(set(required) - set(available))
    if missing:
        raise ValueError(
            f"V2 environment required {kind} assets unavailable for "
            f"{label}: {missing}"
        )


def _asset_id_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    result = tuple(_asset_id(row, label) for row in value)
    if list(result) != sorted(set(result)):
        raise ValueError(f"{label} must be unique and sorted")
    return result


def _coordinate_mode(value: object) -> str:
    result = _nonempty(value, "coordinate_mode")
    if result not in _COORDINATE_MODES:
        raise ValueError("unsupported environment coordinate mode")
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields changed")


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256")
    return value.lower()


def _safe_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} is not a portable identifier")
    return value


def _asset_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty portable string")
    return value


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _exact_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    result = _exact_int(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _positive_int(value: object, label: str) -> int:
    result = _exact_int(value, label)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _optional_radius_fields(
    row: Mapping[str, Any], required: frozenset[str], label: str
) -> None:
    fields = set(row)
    if fields not in (set(required), set(required) | {"radius"}):
        raise ValueError(f"{label} objective fields changed")


def _objective_radius(row: Mapping[str, Any]) -> float:
    return _positive_float(row.get("radius", 1.25), "objective radius")


def _float3(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be one XYZ row")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{label} must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{label} must be finite")
        result.append(number)
    return tuple(result)  # type: ignore[return-value]
