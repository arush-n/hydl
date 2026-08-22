"""Portable, hash-identified surrogate atlas artifacts."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.surrogate.perception import (
    surrogate_entity_visibility_contract,
)
from hytalegym.jax.world.surrogate.spawn_markers import (
    spawn_marker_selection_contract,
)
from hytalegym.jax.world.surrogate.terrain import (
    PRIMARY_ZONE_V1_TERRAIN_SOURCE,
    PROFILE_ONLY_TERRAIN_SOURCE,
    SURROGATE_TERRAIN_SOURCE_SCHEMA,
)
from hytalegym.jax.world.surrogate.types import (
    SurrogateAtlas,
    SurrogateSpawnMarkerCatalog,
    SurrogateTraversalAtlas,
)
from hytalegym.worldgen.region import CHUNK_SIZE
from hytalegym.worldgen.surrogate import (
    SURROGATE_BIOME_PREFAB_LAYER_VERSION,
    SURROGATE_WORLD_SCHEMA,
    SURROGATE_WORLD_VERSION,
    ENTITY_KIND_SPAWN_MARKER,
    SurrogateWorldCapacity,
    entity_type_identity_words,
    estimate_surrogate_atlas_bytes,
)

SURROGATE_ARTIFACT_SCHEMA = "hytalerl_surrogate_world_artifact_v5"
SURROGATE_ARTIFACT_VERSION = 5
_MANIFEST_KEY = "__manifest_json__"
_DIGEST_KEY = "__artifact_sha256__"
_ATLAS_PREFIX = "atlas__"
_SPAWN_MARKER_PREFIX = "spawn_marker__"
_TRAVERSAL_PREFIX = "traversal__"
_PROBABILITY_EPSILON = 1e-12


@dataclass(frozen=True)
class SurrogateWorldArtifact:
    """Immutable rollout arrays plus their complete reproducibility contract."""

    manifest: Mapping[str, Any]
    atlas: SurrogateAtlas
    traversal: SurrogateTraversalAtlas
    spawn_markers: SurrogateSpawnMarkerCatalog

    def __post_init__(self) -> None:
        if not isinstance(self.atlas, SurrogateAtlas):
            raise TypeError("atlas must be a SurrogateAtlas")
        if not isinstance(self.traversal, SurrogateTraversalAtlas):
            raise TypeError("traversal must be a SurrogateTraversalAtlas")
        if not isinstance(self.spawn_markers, SurrogateSpawnMarkerCatalog):
            raise TypeError("spawn_markers must be a SurrogateSpawnMarkerCatalog")
        manifest = _json_copy(self.manifest)
        _validate_manifest(
            manifest,
            self.atlas,
            self.traversal,
            self.spawn_markers,
        )
        object.__setattr__(self, "manifest", manifest)

    @property
    def semantic_sha256(self) -> str:
        digest = hashlib.sha256(_canonical_json(self.manifest).encode())
        for prefix, tree in (
            (_ATLAS_PREFIX, self.atlas),
            (_TRAVERSAL_PREFIX, self.traversal),
            (_SPAWN_MARKER_PREFIX, self.spawn_markers),
        ):
            for name, value in zip(tree._fields, tree, strict=True):
                digest.update(f"{prefix}{name}".encode())
                _update_array_digest(digest, value)
        return digest.hexdigest()

    def save(self, path: str | Path) -> Path:
        """Atomically write compressed numeric arrays without pickle payloads."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            _MANIFEST_KEY: np.asarray(_canonical_json(self.manifest)),
            _DIGEST_KEY: np.asarray(self.semantic_sha256),
        }
        for prefix, tree in (
            (_ATLAS_PREFIX, self.atlas),
            (_TRAVERSAL_PREFIX, self.traversal),
            (_SPAWN_MARKER_PREFIX, self.spawn_markers),
        ):
            payload.update(
                {
                    f"{prefix}{name}": np.asarray(jax.device_get(value))
                    for name, value in zip(tree._fields, tree, strict=True)
                }
            )
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as output:
                temporary = Path(output.name)
                np.savez_compressed(output, **payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "SurrogateWorldArtifact":
        """Load, schema-check, and hash-check a portable artifact."""

        with np.load(Path(path), allow_pickle=False) as archive:
            expected = {
                _MANIFEST_KEY,
                _DIGEST_KEY,
                *(f"{_ATLAS_PREFIX}{name}" for name in SurrogateAtlas._fields),
                *(
                    f"{_TRAVERSAL_PREFIX}{name}"
                    for name in SurrogateTraversalAtlas._fields
                ),
                *(
                    f"{_SPAWN_MARKER_PREFIX}{name}"
                    for name in SurrogateSpawnMarkerCatalog._fields
                ),
            }
            actual = set(archive.files)
            if actual != expected:
                raise ValueError(
                    "surrogate artifact fields differ: "
                    f"missing={sorted(expected - actual)}, "
                    f"extra={sorted(actual - expected)}"
                )
            manifest = json.loads(str(archive[_MANIFEST_KEY].item()))
            declared = str(archive[_DIGEST_KEY].item())
            atlas = SurrogateAtlas(
                *(
                    jnp.asarray(archive[f"{_ATLAS_PREFIX}{name}"].copy())
                    for name in SurrogateAtlas._fields
                )
            )
            traversal = SurrogateTraversalAtlas(
                *(
                    jnp.asarray(archive[f"{_TRAVERSAL_PREFIX}{name}"].copy())
                    for name in SurrogateTraversalAtlas._fields
                )
            )
            spawn_markers = SurrogateSpawnMarkerCatalog(
                *(
                    jnp.asarray(archive[f"{_SPAWN_MARKER_PREFIX}{name}"].copy())
                    for name in SurrogateSpawnMarkerCatalog._fields
                )
            )
        artifact = cls(
            manifest=manifest,
            atlas=atlas,
            traversal=traversal,
            spawn_markers=spawn_markers,
        )
        if declared != artifact.semantic_sha256:
            raise ValueError("surrogate artifact semantic SHA-256 mismatch")
        return artifact


def create_surrogate_world_artifact(
    atlas: SurrogateAtlas,
    traversal: SurrogateTraversalAtlas,
    spawn_markers: SurrogateSpawnMarkerCatalog,
    *,
    contract: Mapping[str, Any],
) -> SurrogateWorldArtifact:
    """Bind validated fixed arrays to one canonical host contract."""

    manifest = {
        "schema": SURROGATE_ARTIFACT_SCHEMA,
        "version": SURROGATE_ARTIFACT_VERSION,
        "authority": "non_authoritative_hytale_informed_surrogate",
        "world_schema": SURROGATE_WORLD_SCHEMA,
        "world_version": SURROGATE_WORLD_VERSION,
        "contract": _json_copy(contract),
        "arrays": _array_specs(atlas, traversal, spawn_markers),
    }
    return SurrogateWorldArtifact(
        manifest=manifest,
        atlas=atlas,
        traversal=traversal,
        spawn_markers=spawn_markers,
    )


def _validate_manifest(
    manifest: dict[str, Any],
    atlas: SurrogateAtlas,
    traversal: SurrogateTraversalAtlas,
    spawn_markers: SurrogateSpawnMarkerCatalog,
) -> None:
    if set(manifest) != {
        "schema",
        "version",
        "authority",
        "world_schema",
        "world_version",
        "contract",
        "arrays",
    }:
        raise ValueError("surrogate artifact manifest fields differ")
    if (
        manifest["schema"] != SURROGATE_ARTIFACT_SCHEMA
        or manifest["version"] != SURROGATE_ARTIFACT_VERSION
        or manifest["authority"] != "non_authoritative_hytale_informed_surrogate"
        or manifest["world_schema"] != SURROGATE_WORLD_SCHEMA
        or manifest["world_version"] != SURROGATE_WORLD_VERSION
    ):
        raise ValueError("unsupported surrogate artifact contract")
    contract = manifest["contract"]
    if not isinstance(contract, dict):
        raise TypeError("surrogate artifact contract must be an object")
    required = {
        "capacity",
        "tile_capacity",
        "active_tile_count",
        "world_id",
        "seed_words",
        "tile_min_xz",
        "actor_profile",
        "terrain",
        "structures",
        "entity_type_palette",
        "spawn_markers",
        "source_assets",
        "capabilities",
        "entity_visibility_contract",
        "unsupported_mechanics",
        "certification",
    }
    if set(contract) != required:
        raise ValueError("surrogate artifact contract fields differ")
    if contract["entity_visibility_contract"] != surrogate_entity_visibility_contract():
        raise ValueError("surrogate entity visibility contract differs")
    capacity_values = contract["capacity"]
    capacity_names = {field.name for field in fields(SurrogateWorldCapacity)}
    if not isinstance(capacity_values, dict) or set(capacity_values) != capacity_names:
        raise ValueError("surrogate artifact capacity fields differ")
    capacity = SurrogateWorldCapacity(**capacity_values)
    tile_capacity = _positive_int(contract["tile_capacity"], "tile_capacity")
    active_tile_count = _positive_int(
        contract["active_tile_count"],
        "active_tile_count",
    )
    if active_tile_count > tile_capacity:
        raise ValueError("active_tile_count exceeds tile_capacity")
    if atlas.tile_mask.shape != (tile_capacity,):
        raise ValueError("atlas tile capacity differs from manifest")
    if traversal.graph_mask.shape != (tile_capacity,):
        raise ValueError("traversal tile capacity differs from manifest")
    active = np.asarray(jax.device_get(atlas.tile_mask), dtype=np.bool_)
    if int(np.count_nonzero(active)) != active_tile_count:
        raise ValueError("active surrogate tile count differs from manifest")
    graph_mask = np.asarray(jax.device_get(traversal.graph_mask), dtype=np.bool_)
    if not np.array_equal(active, graph_mask):
        raise ValueError("atlas and traversal publication masks differ")
    if manifest["arrays"] != _array_specs(atlas, traversal, spawn_markers):
        raise ValueError("surrogate artifact array schema differs")
    if (
        not _all_finite(atlas)
        or not _all_finite(traversal)
        or not _all_finite(spawn_markers)
    ):
        raise ValueError("surrogate artifact contains non-finite arrays")
    estimate = estimate_surrogate_atlas_bytes(
        capacity,
        tile_capacity=tile_capacity,
        environment_capacity=1,
    )
    expected_immutable = (
        estimate.shared_palette_bytes
        + estimate.spawn_marker_catalog_bytes
        + tile_capacity * estimate.immutable_tile_bytes
    )
    actual_immutable = sum(
        np.asarray(jax.device_get(value)).nbytes
        for tree in (atlas, traversal, spawn_markers)
        for value in tree
    )
    if actual_immutable != expected_immutable:
        raise ValueError("surrogate artifact logical memory contract differs")
    _validate_terrain_contract(
        contract["terrain"],
        contract["unsupported_mechanics"],
    )
    _validate_structure_contract(contract["structures"], contract["terrain"])
    _validate_identity(contract, atlas, traversal, active)
    _validate_spawn_markers(contract["spawn_markers"], capacity, atlas, spawn_markers)


def _validate_terrain_contract(
    terrain: object,
    unsupported_mechanics: object,
) -> None:
    required = {
        "profile_sha256",
        "biome_names",
        "config_sha256",
        "semantic_palette_sha256",
        "native_seed_equivalent",
        "source",
    }
    if not isinstance(terrain, dict) or set(terrain) != required:
        raise ValueError("surrogate terrain contract fields differ")
    profile_sha256 = _sha256(terrain["profile_sha256"], "terrain profile SHA-256")
    _sha256(terrain["config_sha256"], "terrain config SHA-256")
    _sha256(
        terrain["semantic_palette_sha256"],
        "terrain semantic palette SHA-256",
    )
    names = terrain["biome_names"]
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(value, str) or not value for value in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("terrain biome_names must be unique non-empty strings")
    if terrain["native_seed_equivalent"] is not False:
        raise ValueError("surrogate terrain cannot claim native seed equivalence")

    source = terrain["source"]
    source_required = {
        "schema",
        "mode",
        "profile_entry_path",
        "profile_sha256",
        "preset_id",
        "preset_version",
        "native_calibrated",
        "fluid_mode",
        "prop_execution_supported",
        "biomes",
    }
    if not isinstance(source, dict) or set(source) != source_required:
        raise ValueError("surrogate terrain source fields differ")
    if source["schema"] != SURROGATE_TERRAIN_SOURCE_SCHEMA:
        raise ValueError("unsupported surrogate terrain source schema")
    _nonempty_string(source["profile_entry_path"], "terrain profile entry path")
    if _sha256(source["profile_sha256"], "terrain source profile SHA-256") != (
        profile_sha256
    ):
        raise ValueError("terrain profile and source hashes differ")
    if source["native_calibrated"] is not False:
        raise ValueError("surrogate terrain source cannot claim native calibration")
    if source["fluid_mode"] != "disabled":
        raise ValueError("surrogate terrain source must keep fluids disabled")
    if source["prop_execution_supported"] is not False:
        raise ValueError("authored biome prop execution is unsupported")
    if (
        not isinstance(unsupported_mechanics, list)
        or "authored_biome_prop_execution" not in unsupported_mechanics
    ):
        raise ValueError("artifact must declare authored biome props unsupported")

    mode = source["mode"]
    biomes = source["biomes"]
    if mode == PROFILE_ONLY_TERRAIN_SOURCE:
        if (
            source["preset_id"] is not None
            or source["preset_version"] != 0
            or biomes != []
        ):
            raise ValueError("profile-only terrain source contains asset bindings")
        return
    if mode != PRIMARY_ZONE_V1_TERRAIN_SOURCE:
        raise ValueError("unsupported surrogate terrain source mode")
    if source["preset_id"] != "primary_zone_v1" or source["preset_version"] != 1:
        raise ValueError("primary-zone terrain preset identity differs")
    if not isinstance(biomes, list) or len(biomes) != len(names):
        raise ValueError("primary-zone terrain sources differ from biome names")

    biome_required = {
        "biome_id",
        "entry_path",
        "content_sha256",
        "prop_graph_sha256",
        "surface_asset_id",
        "subsurface_asset_id",
        "omitted_fluid_asset_ids",
    }
    for expected_name, biome in zip(names, biomes, strict=True):
        if not isinstance(biome, dict) or set(biome) != biome_required:
            raise ValueError("surrogate terrain biome source fields differ")
        if biome["biome_id"] != expected_name:
            raise ValueError("surrogate terrain biome source order differs")
        _nonempty_string(biome["entry_path"], "biome source entry path")
        _sha256(biome["content_sha256"], "biome source SHA-256")
        _nonempty_string(biome["surface_asset_id"], "biome surface asset ID")
        _nonempty_string(
            biome["subsurface_asset_id"],
            "biome subsurface asset ID",
        )
        graph_hashes = biome["prop_graph_sha256"]
        if not isinstance(graph_hashes, list):
            raise TypeError("biome prop graph hashes must be an array")
        for digest in graph_hashes:
            _sha256(digest, "biome prop graph SHA-256")
        omitted_fluids = biome["omitted_fluid_asset_ids"]
        if not isinstance(omitted_fluids, list) or any(
            not isinstance(value, str) or not value for value in omitted_fluids
        ):
            raise ValueError("omitted biome fluids must be non-empty asset IDs")


def _validate_structure_contract(structures: object, terrain: object) -> None:
    if not isinstance(structures, dict):
        raise TypeError("surrogate structures contract must be an object")
    common = {"prefab_sources", "ignored_component_types"}
    legacy = {
        "assignment_entry_path",
        "assignment_sha256",
        "placement",
        *common,
    }
    layered = {
        "mode",
        "source_bundle_sha256",
        "layers",
        *common,
    }
    if set(structures) == legacy:
        _nonempty_string(
            structures["assignment_entry_path"],
            "structure assignment entry path",
        )
        _sha256(structures["assignment_sha256"], "structure assignment SHA-256")
        _validate_structure_placement(structures["placement"], None)
    elif set(structures) == layered:
        if structures["mode"] != "surrogate_biome_prefab_layers_v1":
            raise ValueError("unsupported surrogate structure mode")
        expected_bundle = _sha256(
            structures["source_bundle_sha256"],
            "biome prefab source bundle SHA-256",
        )
        layers = structures["layers"]
        if not isinstance(layers, list) or not layers:
            raise ValueError("biome prefab layers must be a non-empty list")
        layer_fields = {
            "version",
            "mode",
            "biome_id",
            "biome_code",
            "prop_graph_sha256",
            "prop_runtime",
            "assignment_id",
            "assignment_entry_path",
            "assignment_sha256",
            "assignment_root_type",
            "placement",
            "native_distribution",
        }
        identity_keys: list[tuple[int, str, str]] = []
        seen_sources: set[tuple[int, str]] = set()
        code_by_name: dict[str, int] = {}
        name_by_code: dict[int, str] = {}
        probability_by_code: dict[int, float] = {}
        placement_grids: set[tuple[int, int, int]] = set()
        for layer in layers:
            if not isinstance(layer, dict) or set(layer) != layer_fields:
                raise ValueError("biome prefab layer fields differ")
            if (
                layer["version"] != SURROGATE_BIOME_PREFAB_LAYER_VERSION
                or layer["mode"] != "source_bound_surrogate_grid_v1"
                or layer["native_distribution"] is not False
            ):
                raise ValueError("biome prefab layer identity differs")
            name = _nonempty_string(layer["biome_id"], "biome ID")
            code = _nonnegative_int(layer["biome_code"], "biome code")
            if code > np.iinfo(np.uint8).max:
                raise ValueError("biome code exceeds uint8")
            graph = _sha256(
                layer["prop_graph_sha256"],
                "prop graph SHA-256",
            )
            _nonnegative_int(layer["prop_runtime"], "prop runtime")
            assignment = _nonempty_string(
                layer["assignment_id"],
                "assignment ID",
            )
            if any(character in assignment for character in "/\\"):
                raise ValueError("assignment ID must be local")
            assignment_path = _nonempty_string(
                layer["assignment_entry_path"],
                "assignment entry path",
            )
            if not assignment_path.endswith(f"/{assignment}.json"):
                raise ValueError("assignment ID differs from entry path")
            _sha256(layer["assignment_sha256"], "assignment SHA-256")
            if layer["assignment_root_type"] not in {
                "Constant",
                "FieldFunction",
                "Weighted",
            }:
                raise ValueError("unsupported source assignment root type")
            probability = _validate_structure_placement(
                layer["placement"],
                code,
            )
            placement_grids.add(
                (
                    layer["placement"]["grid_spacing"],
                    layer["placement"]["jitter"],
                    layer["placement"]["salt"],
                )
            )
            prior_code = code_by_name.setdefault(name, code)
            prior_name = name_by_code.setdefault(code, name)
            if prior_code != code or prior_name != name:
                raise ValueError("biome prefab layer identity is ambiguous")
            source = (code, graph)
            if source in seen_sources:
                raise ValueError("biome prefab source is duplicated")
            seen_sources.add(source)
            identity_keys.append((code, graph, assignment))
            total = math.fsum((probability_by_code.get(code, 0.0), probability))
            if total > 1.0 + _PROBABILITY_EPSILON:
                raise ValueError("biome prefab placement probabilities exceed one")
            probability_by_code[code] = min(total, 1.0)
        if identity_keys != sorted(identity_keys):
            raise ValueError("biome prefab layers are not canonical")
        if len(placement_grids) != 1:
            raise ValueError("biome prefab layers must share one placement grid")
        actual_bundle = hashlib.sha256(
            json.dumps(layers, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if actual_bundle != expected_bundle:
            raise ValueError("biome prefab source bundle SHA-256 differs")
        _validate_layer_terrain_binding(layers, terrain)
    else:
        raise ValueError("surrogate structures contract fields differ")
    _validate_prefab_sources(structures["prefab_sources"])
    ignored = structures["ignored_component_types"]
    if (
        not isinstance(ignored, list)
        or any(not isinstance(value, str) or not value for value in ignored)
        or ignored != sorted(set(ignored))
    ):
        raise ValueError("ignored component types must be sorted and unique")


def _validate_layer_terrain_binding(
    layers: list[dict[str, Any]],
    terrain: object,
) -> None:
    if not isinstance(terrain, dict):
        raise TypeError("surrogate terrain contract must be an object")
    names = terrain["biome_names"]
    source = terrain["source"]
    source_biomes = source["biomes"]
    for layer in layers:
        code = layer["biome_code"]
        if code >= len(names) or names[code] != layer["biome_id"]:
            raise ValueError("biome prefab layer disagrees with terrain biome order")
        if source["mode"] == PRIMARY_ZONE_V1_TERRAIN_SOURCE:
            biome = source_biomes[code]
            if (
                biome["biome_id"] != layer["biome_id"]
                or layer["prop_graph_sha256"] not in biome["prop_graph_sha256"]
            ):
                raise ValueError("biome prefab source is absent from terrain provenance")


def _validate_structure_placement(
    value: object,
    biome_code: int | None,
) -> float:
    required = {
        "grid_spacing",
        "jitter",
        "placement_probability",
        "salt",
        "allowed_biome_codes",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("structure placement fields differ")
    spacing = _positive_int(value["grid_spacing"], "structure grid spacing")
    jitter = _nonnegative_int(value["jitter"], "structure jitter")
    if 2 * jitter >= spacing:
        raise ValueError("structure jitter exceeds grid spacing")
    probability = value["placement_probability"]
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0.0 <= probability <= 1.0
    ):
        raise ValueError("structure placement probability is invalid")
    _nonnegative_int(value["salt"], "structure placement salt")
    allowed = value["allowed_biome_codes"]
    if biome_code is None:
        if allowed is not None and (
            not isinstance(allowed, list)
            or allowed != sorted(set(allowed))
            or any(
                _nonnegative_int(code, "allowed biome code") > 255
                for code in allowed
            )
        ):
            raise ValueError("allowed biome codes are invalid")
    elif allowed != [biome_code]:
        raise ValueError("biome prefab layer placement mask differs")
    return float(probability)


def _validate_prefab_sources(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("prefab sources must be a non-empty list")
    records: list[tuple[str, str]] = []
    for source in value:
        if not isinstance(source, dict) or set(source) != {"entry_path", "sha256"}:
            raise ValueError("prefab source fields differ")
        records.append(
            (
                _nonempty_string(source["entry_path"], "prefab source path"),
                _sha256(source["sha256"], "prefab source SHA-256"),
            )
        )
    if records != sorted(set(records)):
        raise ValueError("prefab sources must be sorted and unique")


def _validate_identity(
    contract: dict[str, Any],
    atlas: SurrogateAtlas,
    traversal: SurrogateTraversalAtlas,
    active: np.ndarray,
) -> None:
    world_id = _nonnegative_int(contract["world_id"], "world_id")
    atlas_world = np.asarray(jax.device_get(atlas.world_id), dtype=np.int32)
    if not np.all(atlas_world[active] == world_id):
        raise ValueError("atlas world ID differs from manifest")
    seeds = np.asarray(contract["seed_words"], dtype=np.uint32)
    if seeds.shape != (2,) or not np.all(
        np.asarray(jax.device_get(atlas.seed_words))[active] == seeds
    ):
        raise ValueError("atlas seed words differ from manifest")
    origins = np.asarray(contract["tile_min_xz"], dtype=np.int32)
    if origins.shape != (int(np.count_nonzero(active)), 2) or not np.array_equal(
        np.asarray(jax.device_get(atlas.core_min_chunk_xz))[active] - 1,
        origins // CHUNK_SIZE,
    ):
        raise ValueError("atlas tile origins differ from manifest")
    profile = contract["actor_profile"]
    if not isinstance(profile, dict) or set(profile) != {
        "local_bounds",
        "maximum_climb_height",
        "maximum_safe_drop_height",
    }:
        raise ValueError("actor profile fields differ")
    graph_active = np.asarray(jax.device_get(traversal.graph_mask), dtype=np.bool_)
    bounds = np.asarray(profile["local_bounds"], dtype=np.float32)
    if bounds.shape != (6,) or not np.all(
        np.asarray(jax.device_get(traversal.actor_bounds))[graph_active] == bounds
    ):
        raise ValueError("traversal actor bounds differ from manifest")
    climb = float(profile["maximum_climb_height"])
    if not np.all(
        np.asarray(jax.device_get(traversal.maximum_climb_height))[graph_active]
        == np.float32(climb)
    ):
        raise ValueError("traversal climb profile differs from manifest")
    safe_drop = profile["maximum_safe_drop_height"]
    expected_drop_supported = safe_drop is not None
    supported = np.asarray(
        jax.device_get(traversal.drop_safety_supported),
        dtype=np.bool_,
    )
    if not np.all(supported[graph_active] == expected_drop_supported):
        raise ValueError("traversal drop-safety profile differs from manifest")
    expected_drop = 0.0 if safe_drop is None else float(safe_drop)
    if not np.all(
        np.asarray(jax.device_get(traversal.maximum_safe_drop_height))[graph_active]
        == np.float32(expected_drop)
    ):
        raise ValueError("traversal safe-drop height differs from manifest")
    palette = contract["entity_type_palette"]
    if not isinstance(palette, list):
        raise TypeError("entity_type_palette must be an array")
    if int(np.asarray(jax.device_get(atlas.entity_type_count))) != len(palette):
        raise ValueError("entity type palette size differs from atlas")
    identities = np.asarray(
        [value["identity_words"] for value in palette],
        dtype=np.uint32,
    ).reshape(len(palette), 2)
    entity_mask = np.asarray(jax.device_get(atlas.entity_mask), dtype=np.bool_)
    entity_code = np.asarray(jax.device_get(atlas.entity_type_code), dtype=np.uint16)
    entity_identity = np.asarray(
        jax.device_get(atlas.entity_type_identity_words),
        dtype=np.uint32,
    )
    if np.any(entity_code[entity_mask] >= len(palette)) or not np.array_equal(
        entity_identity[entity_mask],
        identities[entity_code[entity_mask]],
    ):
        raise ValueError("entity semantic identity differs from manifest palette")


def _validate_spawn_markers(
    contract: object,
    capacity: SurrogateWorldCapacity,
    atlas: SurrogateAtlas,
    catalog: SurrogateSpawnMarkerCatalog,
) -> None:
    required = {
        "selection_contract",
        "marker_ids",
        "asset_sha256",
        "role_ids",
    }
    if not isinstance(contract, dict) or set(contract) != required:
        raise ValueError("spawn-marker artifact contract fields differ")
    if contract["selection_contract"] != spawn_marker_selection_contract():
        raise ValueError("spawn-marker selection contract differs")

    marker_capacity = capacity.spawn_marker_capacity
    choice_capacity = capacity.spawn_marker_choice_capacity
    expected_shapes = {
        "marker_mask": (marker_capacity,),
        "marker_identity_words": (marker_capacity, 2),
        "asset_sha256_words": (marker_capacity, 8),
        "realtime_respawn": (marker_capacity,),
        "manual_trigger": (marker_capacity,),
        "exclusion_radius": (marker_capacity,),
        "maximum_drop_height": (marker_capacity,),
        "deactivation_distance": (marker_capacity,),
        "deactivation_seconds": (marker_capacity,),
        "choice_mask": (marker_capacity, choice_capacity),
        "role_present": (marker_capacity, choice_capacity),
        "role_identity_words": (marker_capacity, choice_capacity, 2),
        "choice_weight": (marker_capacity, choice_capacity),
        "respawn_seconds": (marker_capacity, choice_capacity),
        "flock_required": (marker_capacity, choice_capacity),
    }
    expected_dtypes = {
        "marker_mask": np.bool_,
        "marker_identity_words": np.uint32,
        "asset_sha256_words": np.uint32,
        "realtime_respawn": np.bool_,
        "manual_trigger": np.bool_,
        "exclusion_radius": np.float32,
        "maximum_drop_height": np.float32,
        "deactivation_distance": np.float32,
        "deactivation_seconds": np.float32,
        "choice_mask": np.bool_,
        "role_present": np.bool_,
        "role_identity_words": np.uint32,
        "choice_weight": np.float32,
        "respawn_seconds": np.float32,
        "flock_required": np.bool_,
    }
    arrays = {
        name: np.asarray(jax.device_get(getattr(catalog, name)))
        for name in catalog._fields
    }
    if any(
        arrays[name].shape != expected_shapes[name]
        or arrays[name].dtype != np.dtype(expected_dtypes[name])
        for name in expected_shapes
    ):
        raise ValueError("spawn-marker artifact arrays differ from capacity")

    marker_ids = contract["marker_ids"]
    asset_sha256 = contract["asset_sha256"]
    role_ids = contract["role_ids"]
    if not all(
        isinstance(value, list) for value in (marker_ids, asset_sha256, role_ids)
    ):
        raise TypeError("spawn-marker provenance must use arrays")
    marker_count = len(marker_ids)
    if (
        marker_count > marker_capacity
        or len(asset_sha256) != marker_count
        or len(role_ids) != marker_count
        or len(set(marker_ids)) != marker_count
        or any(not isinstance(value, str) or not value for value in marker_ids)
    ):
        raise ValueError("spawn-marker provenance sizes or IDs differ")

    expected_marker_mask = np.arange(marker_capacity) < marker_count
    expected_identity = np.zeros((marker_capacity, 2), dtype=np.uint32)
    if marker_count:
        expected_identity[:marker_count] = np.asarray(
            [entity_type_identity_words(value) for value in marker_ids],
            dtype=np.uint32,
        )
        if len({tuple(value) for value in expected_identity[:marker_count]}) != (
            marker_count
        ):
            raise ValueError("spawn-marker identity collision")
    expected_hash = np.zeros((marker_capacity, 8), dtype=np.uint32)
    for index, value in enumerate(asset_sha256):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("spawn-marker asset SHA-256 is malformed")
        expected_hash[index] = np.frombuffer(bytes.fromhex(value), dtype="<u4")

    expected_choice_mask = np.zeros(
        (marker_capacity, choice_capacity),
        dtype=np.bool_,
    )
    expected_role_present = np.zeros_like(expected_choice_mask)
    expected_role_identity = np.zeros(
        (marker_capacity, choice_capacity, 2),
        dtype=np.uint32,
    )
    role_identity_owner: dict[tuple[int, int], str] = {}
    for marker_index, choices in enumerate(role_ids):
        if not isinstance(choices, list) or len(choices) > choice_capacity:
            raise ValueError("spawn-marker role provenance differs")
        expected_choice_mask[marker_index, : len(choices)] = True
        for choice_index, role_id in enumerate(choices):
            if role_id is None:
                continue
            if not isinstance(role_id, str) or not role_id:
                raise ValueError("spawn-marker role ID is malformed")
            expected_role_present[marker_index, choice_index] = True
            identity = entity_type_identity_words(role_id)
            previous = role_identity_owner.setdefault(identity, role_id)
            if previous != role_id:
                raise ValueError("spawn-marker role identity collision")
            expected_role_identity[marker_index, choice_index] = identity

    if (
        not np.array_equal(arrays["marker_mask"], expected_marker_mask)
        or not np.array_equal(arrays["marker_identity_words"], expected_identity)
        or not np.array_equal(arrays["asset_sha256_words"], expected_hash)
        or not np.array_equal(arrays["choice_mask"], expected_choice_mask)
        or not np.array_equal(arrays["role_present"], expected_role_present)
        or not np.array_equal(
            arrays["role_identity_words"],
            expected_role_identity,
        )
    ):
        raise ValueError("spawn-marker arrays differ from provenance")
    if (
        np.any(arrays["exclusion_radius"][:marker_count] < 0.0)
        or np.any(arrays["maximum_drop_height"][:marker_count] <= 0.0)
        or np.any(arrays["deactivation_distance"][:marker_count] <= 0.0)
        or np.any(arrays["deactivation_seconds"][:marker_count] <= 0.0)
        or np.any(arrays["choice_weight"][expected_choice_mask] <= 0.0)
        or np.any(arrays["respawn_seconds"][expected_choice_mask] <= 0.0)
    ):
        raise ValueError("spawn-marker physical metadata is invalid")

    entity_mask = np.asarray(jax.device_get(atlas.entity_mask), dtype=np.bool_)
    entity_kind = np.asarray(jax.device_get(atlas.entity_kind), dtype=np.uint8)
    entity_identity = np.asarray(
        jax.device_get(atlas.entity_type_identity_words),
        dtype=np.uint32,
    )
    placed = entity_identity[entity_mask & (entity_kind == ENTITY_KIND_SPAWN_MARKER)]
    if any(
        np.count_nonzero(np.all(expected_identity[:marker_count] == identity, axis=1))
        != 1
        for identity in placed
    ):
        raise ValueError("placed spawn marker is absent from artifact catalog")


def _array_specs(
    atlas: SurrogateAtlas,
    traversal: SurrogateTraversalAtlas,
    spawn_markers: SurrogateSpawnMarkerCatalog,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for prefix, tree in (
        (_ATLAS_PREFIX, atlas),
        (_TRAVERSAL_PREFIX, traversal),
        (_SPAWN_MARKER_PREFIX, spawn_markers),
    ):
        for name, value in zip(tree._fields, tree, strict=True):
            array = np.asarray(jax.device_get(value))
            if array.dtype.hasobject:
                raise TypeError("surrogate artifact arrays must not use object dtype")
            result[f"{prefix}{name}"] = {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
    return result


def _all_finite(tree: tuple[jax.Array, ...]) -> bool:
    return all(
        not np.issubdtype(array.dtype, np.floating) or np.all(np.isfinite(array))
        for array in (np.asarray(jax.device_get(value)) for value in tree)
    )


def _update_array_digest(digest: Any, value: jax.Array) -> None:
    array = np.ascontiguousarray(jax.device_get(value))
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes())


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("surrogate artifact manifest must be a mapping")
    decoded = json.loads(_canonical_json(dict(value)))
    if not isinstance(decoded, dict):
        raise TypeError("surrogate artifact manifest must be an object")
    return decoded


def _positive_int(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, label: str) -> str:
    result = _nonempty_string(value, label).lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{label} must contain 64 hexadecimal digits")
    return result


__all__ = [
    "SURROGATE_ARTIFACT_SCHEMA",
    "SURROGATE_ARTIFACT_VERSION",
    "SurrogateWorldArtifact",
    "create_surrogate_world_artifact",
]
