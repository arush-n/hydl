"""Publish one authored WorldGen V2 design as a verified JAX environment.

The authored asset pack remains the generation authority.  This module binds
its exact native Region captures, traversal graphs, and structure markers to a
deliberate environment recipe, materializes that recipe once, and writes a
tamper-evident publication manifest.  No terrain is regenerated or inferred in
JAX.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
from typing import Any, Mapping

import jax
import numpy as np

if __package__ and __package__.startswith("experimental."):
    # Normal workspace import, including Arena's publication-backed adapter.
    from ..jax_port.bundles.bundle import (
        CAPABILITY_EXACT_GEOMETRY,
        CAPABILITY_NATIVE_TRAVERSAL,
        CAPABILITY_STRUCTURE_INSTANCES,
        V2JaxBundle,
    )
    from ..jax_port.worlds.environment_recipe import (
        OBJECTIVE_CHECKPOINT,
        OBJECTIVE_REACH,
        V2EnvironmentRecipe,
        create_environment_recipe,
        materialize_environment_recipe,
    )
    from ..jax_port.worlds.minigame_runtime import (
        create_v2_minigame_actors,
        reset_v2_minigame,
        step_v2_minigame,
    )
    from ..jax_port.structures.structure_pack import V2StructurePack
    from ..jax_port.bundles.traversal_pack import V2TraversalPack
else:
    # The CLI also supports direct execution after adding worldgen-v2 to
    # sys.path, where this package is simply ``custom_environments``.
    from jax_port.bundles.bundle import (
        CAPABILITY_EXACT_GEOMETRY,
        CAPABILITY_NATIVE_TRAVERSAL,
        CAPABILITY_STRUCTURE_INSTANCES,
        V2JaxBundle,
    )
    from jax_port.worlds.environment_recipe import (
        OBJECTIVE_CHECKPOINT,
        OBJECTIVE_REACH,
        V2EnvironmentRecipe,
        create_environment_recipe,
        materialize_environment_recipe,
    )
    from jax_port.worlds.minigame_runtime import (
        create_v2_minigame_actors,
        reset_v2_minigame,
        step_v2_minigame,
    )
    from jax_port.structures.structure_pack import V2StructurePack
    from jax_port.bundles.traversal_pack import V2TraversalPack

from .asset_pack import (
    CustomEnvironmentSpec,
    load_environment_spec,
    validate_asset_pack,
)


ENVIRONMENT_BLUEPRINT_SCHEMA = "hytalerl_worldgen_v2_environment_blueprint_v1"
ENVIRONMENT_BLUEPRINT_VERSION = 1
CUSTOM_ENVIRONMENT_PUBLICATION_SCHEMA = (
    "hytalerl_worldgen_v2_custom_environment_publication_v1"
)
CUSTOM_ENVIRONMENT_PUBLICATION_VERSION = 1

_BLUEPRINT_FIELDS = frozenset(
    {
        "schema",
        "version",
        "environment_id",
        "recipe_id",
        "world_structure_asset_id",
        "seeds",
        "required_capabilities",
        "bounds",
        "teams",
        "objective",
        "quality_gates",
        "environment_count",
        "assignment_key",
    }
)
_PUBLICATION_FIELDS = frozenset(
    {
        "schema",
        "version",
        "environment_id",
        "world_structure_asset_id",
        "seeds",
        "source_spec_file_sha256",
        "recipe_blueprint_file_sha256",
        "asset_pack",
        "native_capture",
        "jax_bundle",
        "traversal",
        "structures",
        "environment_recipe",
        "materialization",
        "publication_semantic_sha256",
    }
)
_FILE_REFERENCE_FIELDS = frozenset(
    {"path", "file_sha256", "semantic_sha256"}
)
_REQUIRED_SIDECAR_CAPABILITIES = frozenset(
    {
        CAPABILITY_EXACT_GEOMETRY,
        CAPABILITY_NATIVE_TRAVERSAL,
        CAPABILITY_STRUCTURE_INSTANCES,
    }
)
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class EnvironmentBlueprint:
    """Version-visible intent resolved against immutable captured worlds."""

    path: Path
    file_sha256: str
    environment_id: str
    recipe_id: str
    world_structure_asset_id: str
    seeds: tuple[int, ...]
    required_capabilities: tuple[str, ...]
    bounds: Mapping[str, Any]
    teams: tuple[Mapping[str, Any], ...]
    objective: Mapping[str, Any]
    quality_gates: Mapping[str, Any]
    environment_count: int
    assignment_key: int


def load_environment_blueprint(path: str | Path) -> EnvironmentBlueprint:
    """Load the small, capture-independent environment intent document."""

    source = Path(path).resolve()
    value = _json_object(source)
    _exact_fields(value, _BLUEPRINT_FIELDS, "environment blueprint")
    if value["schema"] != ENVIRONMENT_BLUEPRINT_SCHEMA:
        raise ValueError("unsupported custom environment blueprint schema")
    if _exact_int(value["version"], "blueprint version") != (
        ENVIRONMENT_BLUEPRINT_VERSION
    ):
        raise ValueError("unsupported custom environment blueprint version")
    environment_id = _safe_id(value["environment_id"], "environment_id")
    recipe_id = _safe_id(value["recipe_id"], "recipe_id")
    structure = _nonempty(
        value["world_structure_asset_id"], "world_structure_asset_id"
    )
    raw_seeds = value["seeds"]
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("environment blueprint seeds must be a non-empty list")
    seeds = tuple(_nonnegative_int(seed, "seed") for seed in raw_seeds)
    if list(seeds) != sorted(set(seeds)):
        raise ValueError("environment blueprint seeds must be unique and sorted")
    raw_capabilities = value["required_capabilities"]
    if not isinstance(raw_capabilities, list) or any(
        not isinstance(row, str) for row in raw_capabilities
    ):
        raise TypeError("required_capabilities must be a list of strings")
    capabilities = tuple(raw_capabilities)
    if list(capabilities) != sorted(set(capabilities)):
        raise ValueError("required_capabilities must be unique and sorted")
    missing = _REQUIRED_SIDECAR_CAPABILITIES - set(capabilities)
    if missing:
        raise ValueError(
            "authored environment blueprint omits required exact capabilities: "
            f"{sorted(missing)}"
        )
    bounds = _mapping(value["bounds"], "bounds")
    raw_teams = value["teams"]
    if not isinstance(raw_teams, list) or not raw_teams:
        raise ValueError("environment blueprint teams must be non-empty")
    teams = tuple(_mapping(row, "team") for row in raw_teams)
    objective = _mapping(value["objective"], "objective")
    quality = _mapping(value["quality_gates"], "quality_gates")
    environments = _positive_int(value["environment_count"], "environment_count")
    if environments < len(seeds):
        raise ValueError("environment_count must expose every selected seed")
    return EnvironmentBlueprint(
        path=source,
        file_sha256=_file_sha256(source),
        environment_id=environment_id,
        recipe_id=recipe_id,
        world_structure_asset_id=structure,
        seeds=seeds,
        required_capabilities=capabilities,
        bounds=_json_copy(bounds),
        teams=tuple(_json_copy(row) for row in teams),
        objective=_json_copy(objective),
        quality_gates=_json_copy(quality),
        environment_count=environments,
        assignment_key=_nonnegative_int(value["assignment_key"], "assignment_key"),
    )


def publish_custom_environment(
    publication_path: str | Path,
    *,
    source_spec_path: str | Path,
    blueprint_path: str | Path,
    asset_pack_root: str | Path,
    bundle_path: str | Path,
    capture_root: str | Path,
    traversal_pack_path: str | Path,
    traversal_receipt_path: str | Path,
    structure_pack_path: str | Path,
    structure_receipt_path: str | Path,
    recipe_path: str | Path,
) -> Mapping[str, Any]:
    """Bind, materialize, and publish the complete native-to-JAX chain."""

    destination = Path(publication_path).resolve()
    root = destination.parent
    spec = load_environment_spec(source_spec_path)
    blueprint = load_environment_blueprint(blueprint_path)
    _require_blueprint_matches_spec(blueprint, spec)
    pack_root = Path(asset_pack_root).resolve()
    pack_report = validate_asset_pack(pack_root, spec)

    capture = Path(capture_root).resolve()
    bundle = V2JaxBundle.load(
        bundle_path,
        capture,
        verify_source=True,
        verify_artifacts=True,
    )
    _require_bundle_matches_authoring(bundle, blueprint, spec)
    artifacts = bundle.select(
        blueprint.world_structure_asset_id,
        seeds=blueprint.seeds,
    )

    traversal = V2TraversalPack.load(
        traversal_pack_path,
        bundle,
        verify_graphs=True,
    )
    structures = V2StructurePack.load(
        structure_pack_path,
        bundle,
        verify_snapshots=True,
    )
    if not traversal.covers(artifacts):
        raise ValueError("traversal pack does not cover every authored world")
    if not structures.covers(artifacts):
        raise ValueError("structure pack does not cover every authored world")
    if structures.registry.path.parent.resolve() != pack_root:
        raise ValueError("structure pack does not use the authored pack registry")
    if structures.registry.semantic_sha256 != _registry_semantic(pack_root):
        raise ValueError("structure pack registry differs from authored pack")

    traversal_receipt = _capture_receipt(
        traversal_receipt_path,
        schema="hytalerl_worldgen_v2_traversal_capture_receipt_v1",
        bundle_semantic=bundle.semantic_sha256,
        pack_field="traversal_pack_semantic_sha256",
        pack_semantic=traversal.semantic_sha256,
        artifacts=artifacts,
    )
    structure_receipt = _capture_receipt(
        structure_receipt_path,
        schema="hytalerl_worldgen_v2_structure_capture_receipt_v1",
        bundle_semantic=bundle.semantic_sha256,
        pack_field="structure_pack_semantic_sha256",
        pack_semantic=structures.semantic_sha256,
        artifacts=artifacts,
    )

    recipe = create_environment_recipe(
        recipe_path,
        bundle,
        artifacts,
        recipe_id=blueprint.recipe_id,
        required_capabilities=blueprint.required_capabilities,
        bounds=blueprint.bounds,
        teams=blueprint.teams,
        objective=blueprint.objective,
        quality_gates=blueprint.quality_gates,
    )
    batch = materialize_environment_recipe(
        recipe,
        bundle,
        environment_count=blueprint.environment_count,
        assignment_key=blueprint.assignment_key,
        traversal_pack=traversal,
        structure_pack=structures,
    )
    objective_runtime = _validate_objective_runtime(batch)
    materialization = _materialization_summary(
        batch,
        environment_count=blueprint.environment_count,
        assignment_key=blueprint.assignment_key,
    )
    materialization["objective_runtime"] = objective_runtime
    graph_rows = traversal.load_for_artifacts(artifacts)
    structure_rows = structures.load_for_artifacts(artifacts)

    manifest: dict[str, Any] = {
        "schema": CUSTOM_ENVIRONMENT_PUBLICATION_SCHEMA,
        "version": CUSTOM_ENVIRONMENT_PUBLICATION_VERSION,
        "environment_id": blueprint.environment_id,
        "world_structure_asset_id": blueprint.world_structure_asset_id,
        "seeds": list(blueprint.seeds),
        "source_spec_file_sha256": spec.file_sha256,
        "recipe_blueprint_file_sha256": blueprint.file_sha256,
        "asset_pack": {
            "root": _relative(root, pack_root),
            "pack_semantic_sha256": pack_report["pack_semantic_sha256"],
            "authoring_receipt_semantic_sha256": pack_report[
                "receipt_semantic_sha256"
            ],
        },
        "native_capture": {
            "root": _relative(root, capture),
            "report": _bundle_source_reference(root, capture, bundle, "capture_report"),
            "analysis": _bundle_source_reference(
                root, capture, bundle, "analysis_report"
            ),
            "bridge_jar_sha256": bundle.manifest["source"]["bridge_jar_sha256"],
            "server_jar_sha256": bundle.manifest["source"]["server_jar_sha256"],
            "assets_sha256": bundle.manifest["source"]["assets_sha256"],
        },
        "jax_bundle": _file_reference(
            root, bundle.manifest_path, bundle.semantic_sha256
        ),
        "traversal": {
            "pack": _file_reference(
                root, traversal.manifest_path, traversal.semantic_sha256
            ),
            "capture_receipt": _file_reference(
                root,
                Path(traversal_receipt_path).resolve(),
                traversal_receipt["receipt_semantic_sha256"],
            ),
            "actor_profile": traversal.actor_profile,
            "node_count": sum(row.node_count for row in graph_rows),
            "edge_count": sum(row.edge_count for row in graph_rows),
        },
        "structures": {
            "pack": _file_reference(
                root, structures.manifest_path, structures.semantic_sha256
            ),
            "capture_receipt": _file_reference(
                root,
                Path(structure_receipt_path).resolve(),
                structure_receipt["receipt_semantic_sha256"],
            ),
            "registry_semantic_sha256": structures.registry.semantic_sha256,
            "instance_count": sum(len(row.instances) for row in structure_rows),
            "complete_instance_count": sum(
                instance.capture_coverage == "complete"
                for row in structure_rows
                for instance in row.instances
            ),
            "clipped_instance_count": sum(
                instance.capture_coverage == "clipped_to_capture"
                for row in structure_rows
                for instance in row.instances
            ),
        },
        "environment_recipe": _file_reference(
            root, recipe.path, recipe.semantic_sha256
        ),
        "materialization": materialization,
        "publication_semantic_sha256": "",
    }
    manifest["publication_semantic_sha256"] = _semantic_digest(
        manifest, "publication_semantic_sha256"
    )
    _atomic_json(destination, manifest)
    return load_custom_environment_publication(destination)


def load_custom_environment_publication(path: str | Path) -> Mapping[str, Any]:
    """Validate the publication envelope and every directly referenced file."""

    source = Path(path).resolve()
    value = _json_object(source)
    _exact_fields(value, _PUBLICATION_FIELDS, "custom environment publication")
    if value["schema"] != CUSTOM_ENVIRONMENT_PUBLICATION_SCHEMA:
        raise ValueError("unsupported custom environment publication schema")
    if _exact_int(value["version"], "publication version") != (
        CUSTOM_ENVIRONMENT_PUBLICATION_VERSION
    ):
        raise ValueError("unsupported custom environment publication version")
    semantic = _sha256(
        value["publication_semantic_sha256"], "publication semantic"
    )
    if semantic != _semantic_digest(value, "publication_semantic_sha256"):
        raise ValueError("custom environment publication semantic mismatch")
    root = source.parent
    for reference in (
        value["native_capture"]["report"],
        value["native_capture"]["analysis"],
        value["jax_bundle"],
        value["traversal"]["pack"],
        value["traversal"]["capture_receipt"],
        value["structures"]["pack"],
        value["structures"]["capture_receipt"],
        value["environment_recipe"],
    ):
        _validate_file_reference(root, reference)
    return _json_copy(value)


def validate_custom_environment_publication(
    path: str | Path,
    *,
    source_spec_path: str | Path | None = None,
    blueprint_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Replay every internal provenance check and the fixed-shape JAX load."""

    started = time.perf_counter()
    source = Path(path).resolve()
    manifest = load_custom_environment_publication(source)
    root = source.parent
    environment_id = _safe_id(manifest["environment_id"], "environment_id")
    structure_id = _nonempty(
        manifest["world_structure_asset_id"], "world_structure_asset_id"
    )
    seeds = tuple(
        _nonnegative_int(row, "publication seed") for row in manifest["seeds"]
    )
    if list(seeds) != sorted(set(seeds)) or not seeds:
        raise ValueError("publication seeds must be unique and sorted")

    pack_root = _contained_directory(root, manifest["asset_pack"]["root"])
    authoring_receipt = _validate_authoring_receipt(
        pack_root,
        manifest["asset_pack"],
        environment_id=environment_id,
        world_structure_asset_id=structure_id,
    )
    spec = None
    if source_spec_path is not None:
        spec = load_environment_spec(source_spec_path)
        if spec.file_sha256 != manifest["source_spec_file_sha256"]:
            raise ValueError("publication source spec file changed")
        pack_report = validate_asset_pack(pack_root, spec)
        if pack_report["pack_semantic_sha256"] != manifest["asset_pack"][
            "pack_semantic_sha256"
        ]:
            raise ValueError("publication asset pack semantic changed")

    blueprint = None
    if blueprint_path is not None:
        blueprint = load_environment_blueprint(blueprint_path)
        if blueprint.file_sha256 != manifest["recipe_blueprint_file_sha256"]:
            raise ValueError("publication recipe blueprint file changed")
        if (
            blueprint.environment_id != environment_id
            or blueprint.world_structure_asset_id != structure_id
            or blueprint.seeds != seeds
        ):
            raise ValueError("publication differs from its recipe blueprint")
        if spec is not None:
            _require_blueprint_matches_spec(blueprint, spec)

    capture_root = _contained_directory(root, manifest["native_capture"]["root"])
    bundle_path = _reference_path(root, manifest["jax_bundle"])
    bundle = V2JaxBundle.load(
        bundle_path,
        capture_root,
        verify_source=True,
        verify_artifacts=True,
    )
    if bundle.semantic_sha256 != manifest["jax_bundle"]["semantic_sha256"]:
        raise ValueError("publication JAX bundle semantic changed")
    if spec is not None and blueprint is not None:
        _require_bundle_matches_authoring(bundle, blueprint, spec)
    if authoring_receipt["assets_archive_sha256"] != bundle.manifest["source"][
        "assets_sha256"
    ]:
        raise ValueError("authored pack and captured bundle use different assets")
    artifacts = bundle.select(structure_id, seeds=seeds)

    traversal = V2TraversalPack.load(
        _reference_path(root, manifest["traversal"]["pack"]),
        bundle,
        verify_graphs=True,
    )
    structures = V2StructurePack.load(
        _reference_path(root, manifest["structures"]["pack"]),
        bundle,
        verify_snapshots=True,
    )
    if structures.registry.path.parent.resolve() != pack_root:
        raise ValueError("publication structure registry is outside its asset pack")
    if structures.registry.semantic_sha256 != _registry_semantic(pack_root):
        raise ValueError("publication structure registry differs from its asset pack")
    traversal_receipt = _capture_receipt(
        _reference_path(root, manifest["traversal"]["capture_receipt"]),
        schema="hytalerl_worldgen_v2_traversal_capture_receipt_v1",
        bundle_semantic=bundle.semantic_sha256,
        pack_field="traversal_pack_semantic_sha256",
        pack_semantic=traversal.semantic_sha256,
        artifacts=artifacts,
    )
    structure_receipt = _capture_receipt(
        _reference_path(root, manifest["structures"]["capture_receipt"]),
        schema="hytalerl_worldgen_v2_structure_capture_receipt_v1",
        bundle_semantic=bundle.semantic_sha256,
        pack_field="structure_pack_semantic_sha256",
        pack_semantic=structures.semantic_sha256,
        artifacts=artifacts,
    )
    del traversal_receipt, structure_receipt

    recipe = V2EnvironmentRecipe.load(
        _reference_path(root, manifest["environment_recipe"]), bundle
    )
    if recipe.semantic_sha256 != manifest["environment_recipe"][
        "semantic_sha256"
    ]:
        raise ValueError("publication environment recipe semantic changed")
    if recipe.artifacts != artifacts:
        raise ValueError("publication recipe selects different exact worlds")

    materialization = _mapping(
        manifest["materialization"], "materialization summary"
    )
    environment_count = _positive_int(
        materialization.get("environment_count"), "environment_count"
    )
    assignment_key = _nonnegative_int(
        materialization.get("assignment_key"), "assignment_key"
    )
    if blueprint is not None and (
        environment_count != blueprint.environment_count
        or assignment_key != blueprint.assignment_key
    ):
        raise ValueError("publication materialization differs from blueprint")
    batch = materialize_environment_recipe(
        recipe,
        bundle,
        environment_count=environment_count,
        assignment_key=assignment_key,
        traversal_pack=traversal,
        structure_pack=structures,
    )
    enhanced_objective_evidence = "objective_runtime" in materialization
    objective_runtime = _validate_objective_runtime(
        batch,
        require_native_node_fit=enhanced_objective_evidence,
    )
    observed_materialization = _materialization_summary(
        batch,
        environment_count=environment_count,
        assignment_key=assignment_key,
    )
    if enhanced_objective_evidence:
        observed_materialization["objective_runtime"] = objective_runtime
    else:
        observed_materialization.pop("objective_radius_shape", None)
    if observed_materialization != materialization:
        raise ValueError("publication fixed-shape materialization changed")

    graphs = traversal.load_for_artifacts(artifacts)
    snapshots = structures.load_for_artifacts(artifacts)
    if manifest["traversal"]["node_count"] != sum(
        row.node_count for row in graphs
    ) or manifest["traversal"]["edge_count"] != sum(
        row.edge_count for row in graphs
    ):
        raise ValueError("publication traversal census changed")
    instances = [instance for row in snapshots for instance in row.instances]
    if manifest["structures"]["instance_count"] != len(instances):
        raise ValueError("publication structure instance census changed")
    if manifest["structures"]["complete_instance_count"] != sum(
        row.capture_coverage == "complete" for row in instances
    ) or manifest["structures"]["clipped_instance_count"] != sum(
        row.capture_coverage == "clipped_to_capture" for row in instances
    ):
        raise ValueError("publication structure coverage census changed")
    return {
        "schema": "hytalerl_worldgen_v2_custom_environment_validation_v1",
        "status": "passed",
        "publication_semantic_sha256": manifest[
            "publication_semantic_sha256"
        ],
        "environment_id": environment_id,
        "recipe_semantic_sha256": recipe.semantic_sha256,
        "bundle_semantic_sha256": bundle.semantic_sha256,
        "traversal_pack_semantic_sha256": traversal.semantic_sha256,
        "structure_pack_semantic_sha256": structures.semantic_sha256,
        "exact_world_count": len(artifacts),
        "environment_count": environment_count,
        "traversal_node_count": sum(row.node_count for row in graphs),
        "traversal_edge_count": sum(row.edge_count for row in graphs),
        "structure_instance_count": len(instances),
        "objective_runtime_jit": "passed",
        "validation_seconds": time.perf_counter() - started,
    }


def _require_blueprint_matches_spec(
    blueprint: EnvironmentBlueprint,
    spec: CustomEnvironmentSpec,
) -> None:
    if blueprint.environment_id != spec.environment_id:
        raise ValueError("blueprint and authoring spec name different environments")
    if blueprint.world_structure_asset_id != spec.asset_ids.world_structure:
        raise ValueError("blueprint and authoring spec name different structures")


def _require_bundle_matches_authoring(
    bundle: V2JaxBundle,
    blueprint: EnvironmentBlueprint,
    spec: CustomEnvironmentSpec,
) -> None:
    source = bundle.manifest["source"]
    if source["assets_sha256"] != spec.assets_archive_sha256:
        raise ValueError("captured bundle and authoring spec use different assets")
    if source["server_version"] != spec.server_version:
        raise ValueError("captured bundle and authoring spec use different servers")
    if blueprint.world_structure_asset_id not in bundle.structure_ids:
        raise ValueError("captured bundle omits the authored world structure")


def _capture_receipt(
    path: str | Path,
    *,
    schema: str,
    bundle_semantic: str,
    pack_field: str,
    pack_semantic: str,
    artifacts: tuple[Any, ...],
) -> Mapping[str, Any]:
    value = _json_object(Path(path).resolve())
    if value.get("schema") != schema:
        raise ValueError("unsupported native sidecar capture receipt")
    semantic = _sha256(value.get("receipt_semantic_sha256"), "receipt semantic")
    unsigned = dict(value)
    unsigned.pop("receipt_semantic_sha256", None)
    if semantic != _canonical_sha256(unsigned):
        raise ValueError("native sidecar capture receipt semantic mismatch")
    if value.get("source_bundle_semantic_sha256") != bundle_semantic:
        raise ValueError("native sidecar receipt names another JAX bundle")
    if value.get(pack_field) != pack_semantic:
        raise ValueError("native sidecar receipt names another sidecar pack")
    rows = value.get("rows")
    if not isinstance(rows, list) or value.get("artifact_count") != len(rows):
        raise ValueError("native sidecar receipt artifact count changed")
    expected = {
        (row.structure, row.seed, row.region_semantic_sha256) for row in artifacts
    }
    observed = {
        (
            row.get("structure"),
            row.get("seed"),
            row.get("source_region_semantic_sha256"),
        )
        for row in rows
        if isinstance(row, Mapping)
        and row.get("live_region_verification") == "semantic_sha256_equal"
    }
    if not expected.issubset(observed):
        raise ValueError("native sidecar receipt lacks exact live Region evidence")
    return value


def _materialization_summary(
    batch: Any,
    *,
    environment_count: int,
    assignment_key: int,
) -> dict[str, Any]:
    world_ids = np.asarray(jax.device_get(batch.worlds.environment_world_id))
    counts = np.bincount(world_ids, minlength=len(batch.worlds.artifacts))
    if world_ids.shape != (environment_count,):
        raise ValueError("materialized environment batch shape changed")
    if np.any(counts == 0):
        raise ValueError("materialized environment batch omits an authored seed")
    if batch.worlds.traversal_atlas is None:
        raise ValueError("materialized environment lacks native traversal")
    if batch.worlds.structure_instance_atlas is None:
        raise ValueError("materialized environment lacks structure instances")
    return {
        "environment_count": environment_count,
        "assignment_key": assignment_key,
        "exact_world_count": len(batch.worlds.artifacts),
        "environment_world_counts": {
            str(row.seed): int(count)
            for row, count in zip(batch.worlds.artifacts, counts, strict=True)
        },
        "physical_cell_code_shape": list(batch.worlds.physical_atlas.cell_code.shape),
        "traversal_node_shape": list(
            batch.worlds.traversal_atlas.node_mask.shape
        ),
        "traversal_edge_shape": list(
            batch.worlds.traversal_atlas.edge_mask.shape
        ),
        "structure_instance_shape": list(
            batch.worlds.structure_instance_atlas.instance_mask.shape
        ),
        "team_spawn_shape": list(batch.runtime.team_spawn_position.shape),
        "objective_radius_shape": list(batch.runtime.objective_radius.shape),
    }


def _validate_objective_runtime(
    batch: Any,
    *,
    require_native_node_fit: bool = True,
) -> dict[str, Any]:
    """Compile neutral and nearest-native-node objective steps for the batch."""

    actors = create_v2_minigame_actors(
        batch.runtime.team_spawn_position,
        batch.runtime.team_key,
        batch.runtime.team_mask,
        batch.runtime.team_mask,
    )
    state = jax.jit(reset_v2_minigame)(batch.runtime)
    result = jax.jit(step_v2_minigame)(batch.runtime, state, actors)
    reward = np.asarray(jax.device_get(result.reward))
    terminated = np.asarray(jax.device_get(result.terminated))
    truncated = np.asarray(jax.device_get(result.truncated))
    expected_team_shape = tuple(batch.runtime.team_mask.shape)
    expected_batch_shape = tuple(batch.runtime.objective_kind.shape)
    if reward.shape != expected_team_shape or not np.all(np.isfinite(reward)):
        raise ValueError("V2 objective runtime returned invalid team rewards")
    if terminated.shape != expected_batch_shape:
        raise ValueError("V2 objective runtime returned invalid termination shape")
    if truncated.shape != expected_batch_shape or np.any(truncated):
        raise ValueError("V2 objective runtime returned an authored truncation")
    if not require_native_node_fit:
        return {"neutral_spawn_step": "passed"}

    traversal = batch.worlds.traversal_atlas
    if traversal is None:
        raise ValueError("V2 objective runtime lacks native traversal")
    world_ids = np.asarray(
        jax.device_get(batch.worlds.environment_world_id), dtype=np.int32
    )
    objective_kind = np.asarray(
        jax.device_get(batch.runtime.objective_kind), dtype=np.uint8
    )
    targets = np.asarray(
        jax.device_get(batch.runtime.objective_position[:, 0]),
        dtype=np.float32,
    )
    radii = np.asarray(
        jax.device_get(batch.runtime.objective_radius), dtype=np.float32
    )
    node_positions = np.asarray(
        jax.device_get(traversal.node_position), dtype=np.float32
    )
    node_masks = np.asarray(jax.device_get(traversal.node_mask), dtype=np.bool_)
    actor_positions = np.asarray(
        jax.device_get(batch.runtime.team_spawn_position), dtype=np.float32
    ).copy()
    nearest_distances = np.zeros(expected_batch_shape, dtype=np.float32)
    navigation = np.isin(
        objective_kind, (OBJECTIVE_REACH, OBJECTIVE_CHECKPOINT)
    )
    for environment_index in np.flatnonzero(navigation):
        world_id = int(world_ids[environment_index])
        available = node_positions[world_id, node_masks[world_id]]
        if available.shape[0] == 0:
            raise ValueError("V2 navigation objective has no native nodes")
        squared = np.sum(
            (available - targets[environment_index]) ** 2,
            axis=1,
        )
        nearest = available[int(np.argmin(squared))]
        nearest_distances[environment_index] = float(np.sqrt(np.min(squared)))
        actor_positions[environment_index, :, :] = nearest
    if np.any(nearest_distances[navigation] > radii[navigation] + 1e-5):
        raise ValueError("V2 objective radius misses the nearest native node")
    native_node_actors = create_v2_minigame_actors(
        actor_positions,
        batch.runtime.team_key,
        batch.runtime.team_mask,
        batch.runtime.team_mask,
    )
    native_node_result = jax.jit(step_v2_minigame)(
        batch.runtime,
        jax.jit(reset_v2_minigame)(batch.runtime),
        native_node_actors,
    )
    completed = np.asarray(
        jax.device_get(native_node_result.objective_completed), dtype=np.bool_
    )
    advanced = np.asarray(
        jax.device_get(native_node_result.checkpoint_advanced), dtype=np.bool_
    )
    reach = objective_kind == OBJECTIVE_REACH
    checkpoint = objective_kind == OBJECTIVE_CHECKPOINT
    if np.any(reach & ~completed):
        raise ValueError("V2 reach objective failed at its nearest native node")
    if np.any(checkpoint & ~np.any(advanced, axis=1)):
        raise ValueError("V2 checkpoint failed at its nearest native node")
    return {
        "neutral_spawn_step": "passed",
        "nearest_native_node_step": "passed",
        "navigation_environment_count": int(np.count_nonzero(navigation)),
        "reach_completed_count": int(np.count_nonzero(reach & completed)),
        "checkpoint_advanced_count": int(
            np.count_nonzero(checkpoint & np.any(advanced, axis=1))
        ),
        "nearest_native_node_distance_maximum": (
            float(np.max(nearest_distances[navigation]))
            if np.any(navigation)
            else 0.0
        ),
        "authored_radius_minimum": (
            float(np.min(radii[navigation])) if np.any(navigation) else 0.0
        ),
    }


def _registry_semantic(pack_root: Path) -> str:
    value = _json_object(pack_root / "structure-registry.registry")
    return _sha256(value.get("registry_semantic_sha256"), "registry semantic")


def _bundle_source_reference(
    publication_root: Path,
    capture_root: Path,
    bundle: V2JaxBundle,
    field: str,
) -> dict[str, str]:
    source = bundle.manifest["source"][field]
    path = capture_root / PurePosixPath(source["path"])
    return _file_reference(
        publication_root,
        path,
        str(source["semantic_sha256"]),
    )


def _file_reference(root: Path, path: Path, semantic: str) -> dict[str, str]:
    source = path.resolve()
    return {
        "path": _relative(root, source),
        "file_sha256": _file_sha256(source),
        "semantic_sha256": _sha256(semantic, "file semantic"),
    }


def _validate_file_reference(root: Path, value: object) -> None:
    row = _mapping(value, "file reference")
    _exact_fields(row, _FILE_REFERENCE_FIELDS, "file reference")
    path = _contained(root, row["path"])
    if _file_sha256(path) != _sha256(row["file_sha256"], "referenced file"):
        raise ValueError(f"custom environment referenced file changed: {path}")
    _sha256(row["semantic_sha256"], "referenced semantic")


def _reference_path(root: Path, value: object) -> Path:
    _validate_file_reference(root, value)
    return _contained(root, _mapping(value, "file reference")["path"])


def _validate_authoring_receipt(
    pack_root: Path,
    publication_value: object,
    *,
    environment_id: str,
    world_structure_asset_id: str,
) -> Mapping[str, Any]:
    published = _mapping(publication_value, "published asset pack")
    receipt_path = pack_root / "authoring-receipt.json"
    receipt = _json_object(receipt_path)
    semantic = _sha256(
        receipt.get("receipt_semantic_sha256"), "authoring receipt semantic"
    )
    if semantic != _semantic_digest(receipt, "receipt_semantic_sha256"):
        raise ValueError("authored pack receipt semantic mismatch")
    if semantic != published.get("authoring_receipt_semantic_sha256"):
        raise ValueError("publication names another authoring receipt")
    if receipt.get("environment_id") != environment_id:
        raise ValueError("authored pack receipt names another environment")
    if receipt.get("world_structure_asset_id") != world_structure_asset_id:
        raise ValueError("authored pack receipt names another world structure")
    output_files = _mapping(receipt.get("output_files"), "authored output files")
    actual = {
        path.relative_to(pack_root).as_posix()
        for path in pack_root.rglob("*")
        if path.is_file()
    }
    if actual != set(output_files) | {"authoring-receipt.json"}:
        raise ValueError("authored pack file set changed")
    normalized: dict[str, str] = {}
    for relative, expected in output_files.items():
        path = _contained(pack_root, relative)
        digest = _sha256(expected, f"authored output {relative}")
        if _file_sha256(path) != digest:
            raise ValueError(f"authored pack output changed: {relative}")
        normalized[str(relative)] = digest
    pack_semantic = _canonical_sha256(normalized)
    if pack_semantic != receipt.get("pack_semantic_sha256") or (
        pack_semantic != published.get("pack_semantic_sha256")
    ):
        raise ValueError("authored pack semantic changed")
    return receipt


def _relative(root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("publication artifacts must share one operational root") from error
    if not relative.parts:
        raise ValueError("publication reference cannot name its root directory")
    return relative.as_posix()


def _contained(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("publication paths must be non-empty POSIX paths")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("publication path escapes its root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"publication referenced file is unavailable: {value}")
    return path


def _contained_directory(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("publication directory must be a non-empty POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("publication directory escapes its root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_dir():
        raise ValueError(f"publication directory is unavailable: {value}")
    return path


def _semantic_digest(value: Mapping[str, Any], field: str) -> str:
    stable = dict(value)
    stable[field] = ""
    return _canonical_sha256(stable)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key!r}: {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


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
    return value


def _safe_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} is not a portable identifier")
    return value


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


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


__all__ = [
    "CUSTOM_ENVIRONMENT_PUBLICATION_SCHEMA",
    "CUSTOM_ENVIRONMENT_PUBLICATION_VERSION",
    "ENVIRONMENT_BLUEPRINT_SCHEMA",
    "ENVIRONMENT_BLUEPRINT_VERSION",
    "EnvironmentBlueprint",
    "load_custom_environment_publication",
    "load_environment_blueprint",
    "publish_custom_environment",
    "validate_custom_environment_publication",
]
