"""Immutable catalog compiler for exact WorldGen V2 JAX worlds.

The bundle deliberately does not run Hytale generation.  It compiles an
already validated V2 capture report into a small, direct lookup catalog whose
artifact paths remain relative to a caller-supplied capture root.  Runtime
loaders can therefore hash and decompress only selected worlds instead of
walking the complete source corpus on every process start.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import operator
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping, Sequence


BUNDLE_SCHEMA = "hytalerl_worldgen_v2_jax_bundle_v1"
BUNDLE_VERSION = 1
BUNDLE_AUTHORITY = "exact_captured_worldgen_v2"
BUNDLE_RUNTIME_SOURCE = "artifact_only_no_live_generation"
BUNDLE_ASSIGNMENT_METHOD = "sha256_balanced_cycle_v1"

CAPABILITY_EXACT_GEOMETRY = "exact_region_geometry"
CAPABILITY_STABLE_BLOCKS = "stable_block_identity"
CAPABILITY_STABLE_FLUIDS = "stable_fluid_identity"
CAPABILITY_STATIC_SPAWN = "static_spawn_position"
CAPABILITY_NATIVE_TRAVERSAL = "native_traversal"
CAPABILITY_STRUCTURE_INSTANCES = "structure_instance_identity"
CAPABILITY_BIOME_IDENTITY = "biome_identity"
CAPABILITY_DYNAMIC_ENTITIES = "dynamic_entities"
CAPABILITY_NATIVE_LIGHTING = "native_lighting"

_CAPABILITIES = (
    CAPABILITY_EXACT_GEOMETRY,
    CAPABILITY_STABLE_BLOCKS,
    CAPABILITY_STABLE_FLUIDS,
    CAPABILITY_STATIC_SPAWN,
    CAPABILITY_NATIVE_TRAVERSAL,
    CAPABILITY_STRUCTURE_INSTANCES,
    CAPABILITY_BIOME_IDENTITY,
    CAPABILITY_DYNAMIC_ENTITIES,
    CAPABILITY_NATIVE_LIGHTING,
)
_AVAILABLE_CAPABILITIES = frozenset(
    {
        CAPABILITY_EXACT_GEOMETRY,
        CAPABILITY_STABLE_BLOCKS,
        CAPABILITY_STABLE_FLUIDS,
        CAPABILITY_STATIC_SPAWN,
    }
)
_CAPABILITY_REASON = {
    CAPABILITY_EXACT_GEOMETRY: "complete_exact_region_snapshot",
    CAPABILITY_STABLE_BLOCKS: "complete_region_block_semantic_sidecar",
    CAPABILITY_STABLE_FLUIDS: "complete_region_fluid_getId_sidecar",
    CAPABILITY_STATIC_SPAWN: "native_reset_spawn_position",
    CAPABILITY_NATIVE_TRAVERSAL: "traversal_sidecars_not_compiled",
    CAPABILITY_STRUCTURE_INSTANCES: "capture_has_geometry_not_instance_labels",
    CAPABILITY_BIOME_IDENTITY: "capture_protocol_has_no_per_cell_biome_id",
    CAPABILITY_DYNAMIC_ENTITIES: "region_capture_is_static_world_geometry",
    CAPABILITY_NATIVE_LIGHTING: "v2_corpus_has_no_native_light_sidecars",
}

_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "version",
        "authority",
        "runtime_source",
        "bundle_id",
        "source",
        "measurement_split_contract",
        "capabilities",
        "artifact_count",
        "structure_count",
        "structures",
        "bundle_semantic_sha256",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "capture_report",
        "analysis_report",
        "capture_plan",
        "world_template",
        "worldgen_provider",
        "server_version",
        "bridge_jar_sha256",
        "server_jar_sha256",
        "assets_sha256",
    }
)
_FILE_REF_FIELDS = frozenset({"path", "file_sha256", "semantic_sha256"})
_SPLIT_FIELDS = frozenset({"method", "salt_sha256", "scope", "mapping"})
_CAPABILITY_FIELDS = frozenset({"available", "reason"})
_STRUCTURE_FIELDS = frozenset(
    {
        "structure",
        "artifact_count",
        "region_library",
        "block_semantic_library",
        "traversal",
        "artifacts",
    }
)
_LIBRARY_FIELDS = frozenset(
    {"manifest_path", "manifest_file_sha256", "library_semantic_sha256"}
)
_TRAVERSAL_FIELDS = frozenset({"available", "reason"})
_ARTIFACT_FIELDS = frozenset(
    {
        "structure",
        "seed",
        "split",
        "core_min_chunk_xz",
        "region",
        "block_semantics",
        "fluid_semantics",
        "spawn_position",
        "features",
    }
)
_FEATURE_FIELDS = frozenset(
    {
        "terrain_relief_blocks",
        "surface_height_minimum",
        "surface_height_median",
        "surface_height_maximum",
        "surface_step_maximum",
        "surface_step_mean",
        "surface_step_nonzero_fraction",
        "subsurface_void_fraction",
        "unique_non_air_material_ids",
        "non_air_material_asset_ids",
        "core_fluid_cells",
        "full_fluid_cells",
        "fluid_asset_ids",
        "damaging_flag_cells",
        "maximum_block_damage",
        "maximum_fluid_damage",
    }
)
_SPLITS = frozenset({"calibration", "heldout"})
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_SAFE_BUNDLE_ID = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class V2JaxArtifact:
    """One directly addressable exact V2 world and its compact features."""

    structure: str
    seed: int
    split: str
    core_min_chunk_xz: tuple[int, int]
    region_path: str
    region_file_sha256: str
    region_semantic_sha256: str
    block_semantic_path: str
    block_semantic_file_sha256: str
    block_semantic_sha256: str
    fluid_semantic_path: str
    fluid_semantic_file_sha256: str
    fluid_semantic_sha256: str
    spawn_position: tuple[float, float, float]
    features: Mapping[str, Any]


@dataclass(frozen=True)
class V2JaxBundle:
    """Validated lightweight index over an external exact capture root."""

    manifest_path: Path
    artifact_root: Path
    manifest: Mapping[str, Any]
    artifacts: tuple[V2JaxArtifact, ...]

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        artifact_root: str | Path,
        *,
        verify_source: bool = True,
        verify_artifacts: bool = False,
    ) -> "V2JaxBundle":
        path = Path(manifest_path).resolve()
        root = Path(artifact_root).resolve()
        if not root.is_dir():
            raise ValueError(f"artifact root does not exist: {root}")
        value = _json_object(path)
        _exact_fields(value, _MANIFEST_FIELDS, "V2 JAX bundle")
        if value["schema"] != BUNDLE_SCHEMA:
            raise ValueError("unsupported V2 JAX bundle schema")
        if _exact_int(value["version"], "bundle version") != BUNDLE_VERSION:
            raise ValueError("unsupported V2 JAX bundle version")
        if value["authority"] != BUNDLE_AUTHORITY:
            raise ValueError("V2 JAX bundle authority changed")
        if value["runtime_source"] != BUNDLE_RUNTIME_SOURCE:
            raise ValueError("V2 JAX bundle must reject live generation")
        _bundle_id(value["bundle_id"])
        expected_semantic = _bundle_digest(value)
        if _sha256(value["bundle_semantic_sha256"], "bundle semantic") != (
            expected_semantic
        ):
            raise ValueError("V2 JAX bundle semantic SHA-256 mismatch")

        source = _mapping(value["source"], "bundle source")
        _exact_fields(source, _SOURCE_FIELDS, "bundle source")
        for field in ("capture_report", "analysis_report", "capture_plan"):
            _validate_file_ref(source[field], f"source {field}")
        if source["world_template"] != "hytale_generator":
            raise ValueError("bundle source is not the explicit V2 world")
        if source["worldgen_provider"] != "HytaleGenerator":
            raise ValueError("bundle source provider changed")
        _nonempty(source["server_version"], "server_version")
        for field in ("bridge_jar_sha256", "server_jar_sha256", "assets_sha256"):
            _sha256(source[field], field)

        split_contract = _validate_split_contract(
            value["measurement_split_contract"]
        )
        capabilities = _validate_capabilities(value["capabilities"])
        if capabilities != _AVAILABLE_CAPABILITIES:
            raise ValueError("V2 JAX bundle capability availability changed")

        structures = value["structures"]
        structure_count = _positive_int(
            value["structure_count"], "structure_count"
        )
        artifact_count = _positive_int(
            value["artifact_count"], "artifact_count"
        )
        if not isinstance(structures, list) or len(structures) != structure_count:
            raise ValueError("bundle structures differ from structure_count")
        parsed: list[V2JaxArtifact] = []
        seen_structures: set[str] = set()
        for structure_value in structures:
            parsed.extend(
                _parse_structure(
                    structure_value,
                    split_contract=split_contract,
                    seen_structures=seen_structures,
                )
            )
        if len(parsed) != artifact_count:
            raise ValueError("bundle artifacts differ from artifact_count")
        identities = [(row.structure, row.seed) for row in parsed]
        if len(set(identities)) != len(identities):
            raise ValueError("bundle structure/seed identities must be unique")
        semantics = [row.region_semantic_sha256 for row in parsed]
        if len(set(semantics)) != len(semantics):
            raise ValueError("bundle Region semantics must be unique")

        if verify_source:
            _verify_source_files(root, source)
        if verify_artifacts:
            for row in parsed:
                for relative, expected in (
                    (row.region_path, row.region_file_sha256),
                    (row.block_semantic_path, row.block_semantic_file_sha256),
                    (row.fluid_semantic_path, row.fluid_semantic_file_sha256),
                ):
                    _require_file_sha256(root, relative, expected)
        return cls(
            manifest_path=path,
            artifact_root=root,
            manifest=_json_copy(value),
            artifacts=tuple(parsed),
        )

    @property
    def semantic_sha256(self) -> str:
        return str(self.manifest["bundle_semantic_sha256"])

    @property
    def structure_ids(self) -> tuple[str, ...]:
        return tuple(row["structure"] for row in self.manifest["structures"])

    @property
    def available_capabilities(self) -> frozenset[str]:
        return frozenset(
            name
            for name, value in self.manifest["capabilities"].items()
            if value["available"] is True
        )

    def require_capabilities(self, names: Sequence[str]) -> None:
        if isinstance(names, (str, bytes)):
            raise TypeError("capability names must be a sequence")
        unknown = set(names) - set(_CAPABILITIES)
        if unknown:
            raise ValueError(f"unknown V2 JAX capabilities: {sorted(unknown)}")
        missing = set(names) - set(self.available_capabilities)
        if missing:
            reasons = {
                name: self.manifest["capabilities"][name]["reason"]
                for name in sorted(missing)
            }
            raise ValueError(f"required V2 JAX capabilities unavailable: {reasons}")

    def select(
        self,
        structure: str,
        *,
        split: str | None = None,
        seeds: Sequence[int] | None = None,
        count: int | None = None,
        selection_key: int = 0,
    ) -> tuple[V2JaxArtifact, ...]:
        """Select exact worlds using the global cross-structure seed split."""

        selected_structure = _structure_id(structure)
        candidates = [
            row for row in self.artifacts if row.structure == selected_structure
        ]
        if not candidates:
            raise ValueError(f"unknown V2 structure: {selected_structure}")
        if split is not None:
            selected_split = _split(split)
            candidates = [row for row in candidates if row.split == selected_split]
        if seeds is not None:
            if isinstance(seeds, (str, bytes)):
                raise TypeError("seeds must be a sequence")
            requested = tuple(_nonnegative_int(value, "seed") for value in seeds)
            if len(set(requested)) != len(requested):
                raise ValueError("requested seeds must be unique")
            by_seed = {row.seed: row for row in candidates}
            missing = [seed for seed in requested if seed not in by_seed]
            if missing:
                raise ValueError(
                    f"requested seeds unavailable for {selected_structure}: {missing}"
                )
            candidates = [by_seed[seed] for seed in requested]
        else:
            key = _nonnegative_int(selection_key, "selection_key")
            candidates.sort(
                key=lambda row: (
                    hashlib.sha256(
                        (
                            f"{BUNDLE_ASSIGNMENT_METHOD}\0{key}\0"
                            f"{row.region_semantic_sha256}"
                        ).encode()
                    ).digest(),
                    row.region_semantic_sha256,
                )
            )
        if count is not None:
            requested_count = _positive_int(count, "selection count")
            if requested_count > len(candidates):
                raise ValueError("selection count exceeds available worlds")
            candidates = candidates[:requested_count]
        if not candidates:
            raise ValueError("V2 JAX selection is empty")
        return tuple(candidates)


def compile_capture_bundle(
    capture_root: str | Path,
    destination: str | Path,
    *,
    bundle_id: str | None = None,
    verify_artifact_hashes: bool = True,
) -> V2JaxBundle:
    """Compile one analyzed exact V2 capture into a direct JAX catalog."""

    root = Path(capture_root).resolve()
    if not root.is_dir():
        raise ValueError(f"capture root does not exist: {root}")
    report_path = root / "report.json"
    analysis_path = root / "analysis.json"
    report = _json_object(report_path)
    analysis = _json_object(analysis_path)
    _require_report(report)
    _require_analysis(analysis, report_path, report)

    plan_relative = _relative_path(report.get("capture_plan"))
    plan_path = _artifact_path(root, plan_relative)
    plan = _json_object(plan_path)
    plan_semantic = _canonical_sha256(plan)
    if _sha256(report.get("capture_plan_sha256"), "capture plan semantic") != (
        plan_semantic
    ):
        raise ValueError("capture plan semantic SHA-256 mismatch")
    _require_plan(plan, report)

    split_mapping = {
        str(seed): _split(value)
        for seed, value in _mapping(
            report["measurement_split_contract"].get("mapping"),
            "measurement split mapping",
        ).items()
    }
    if split_mapping != {
        str(seed): _split(value) for seed, value in plan["seed_split"].items()
    }:
        raise ValueError("capture report and plan seed splits differ")
    split_contract = {
        "method": str(report["measurement_split_contract"].get("method")),
        "salt_sha256": _sha256(plan["split_salt_sha256"], "split salt"),
        "scope": str(report["measurement_split_contract"].get("scope")),
        "mapping": split_mapping,
    }
    _validate_split_contract(split_contract)

    analysis_rows = _analysis_artifact_index(analysis)
    compiled_structures: list[dict[str, Any]] = []
    all_artifact_count = 0
    expected_structures = tuple(_structure_id(value) for value in plan["structures"])
    observed_structures = tuple(
        _structure_id(value.get("structure")) for value in report["structures"]
    )
    if observed_structures != expected_structures:
        raise ValueError("capture report structures differ from frozen plan")

    for structure_row in report["structures"]:
        compiled = _compile_structure(
            root,
            structure_row,
            split_mapping=split_mapping,
            analysis_rows=analysis_rows,
            verify_artifact_hashes=verify_artifact_hashes,
        )
        compiled_structures.append(compiled)
        all_artifact_count += int(compiled["artifact_count"])
    if all_artifact_count != _positive_int(
        report["artifact_count"], "capture artifact_count"
    ):
        raise ValueError("compiled artifact count differs from capture report")

    selected_bundle_id = _bundle_id(
        root.name if bundle_id is None else bundle_id
    )
    source = {
        "capture_report": _file_ref(
            root,
            "report.json",
            semantic_sha256=report["report_semantic_sha256"],
        ),
        "analysis_report": _file_ref(
            root,
            "analysis.json",
            semantic_sha256=analysis["report_semantic_sha256"],
        ),
        "capture_plan": _file_ref(
            root,
            plan_relative,
            semantic_sha256=plan_semantic,
        ),
        "world_template": report["world_template"],
        "worldgen_provider": report["worldgen_provider"],
        "server_version": str(
            report["structures"][0]["artifacts"][0].get(
                "server_version",
                plan.get("server_version", "0.5.7"),
            )
        ),
        "bridge_jar_sha256": _sha256(
            plan["bridge_jar_sha256"], "bridge JAR"
        ),
        "server_jar_sha256": _sha256(
            plan["server_jar_sha256"], "server JAR"
        ),
        "assets_sha256": _sha256(plan["assets_sha256"], "assets"),
    }
    # Server version is carried authoritatively by every Region library.
    source["server_version"] = str(
        _json_object(
            _artifact_path(
                root,
                compiled_structures[0]["region_library"]["manifest_path"],
            )
        )["capture_contract"]["server_version"]
    )
    manifest: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "version": BUNDLE_VERSION,
        "authority": BUNDLE_AUTHORITY,
        "runtime_source": BUNDLE_RUNTIME_SOURCE,
        "bundle_id": selected_bundle_id,
        "source": source,
        "measurement_split_contract": split_contract,
        "capabilities": _capability_manifest(),
        "artifact_count": all_artifact_count,
        "structure_count": len(compiled_structures),
        "structures": compiled_structures,
        "bundle_semantic_sha256": "",
    }
    manifest["bundle_semantic_sha256"] = _bundle_digest(manifest)
    destination_path = Path(destination).resolve()
    _atomic_json(destination_path, manifest)
    return V2JaxBundle.load(
        destination_path,
        root,
        verify_source=True,
        verify_artifacts=False,
    )


def _compile_structure(
    root: Path,
    structure_row: Mapping[str, Any],
    *,
    split_mapping: Mapping[str, str],
    analysis_rows: Mapping[tuple[str, int], Mapping[str, Any]],
    verify_artifact_hashes: bool,
) -> dict[str, Any]:
    structure = _structure_id(structure_row.get("structure"))
    directory = _relative_path(structure_row.get("directory"))
    structure_report_relative = _relative_path(f"{directory}/report.json")
    structure_report = _json_object(_artifact_path(root, structure_report_relative))
    if structure_report != structure_row:
        raise ValueError(f"nested and on-disk structure reports differ: {structure}")
    if _report_digest(structure_report) != _sha256(
        structure_report.get("report_semantic_sha256"),
        f"{structure} report semantic",
    ):
        raise ValueError(f"structure report semantic differs: {structure}")

    region_manifest_relative = _relative_path(
        structure_row.get("region_library_manifest")
    )
    block_manifest_relative = _relative_path(
        structure_row.get("block_semantic_library_manifest")
    )
    region_manifest = _json_object(_artifact_path(root, region_manifest_relative))
    block_manifest = _json_object(_artifact_path(root, block_manifest_relative))
    _require_library_manifest(
        region_manifest,
        schema="hytalerl_region_artifact_library_v1",
        expected_semantic=structure_row.get("region_library_semantic_sha256"),
        label=f"{structure} Region library",
    )
    _require_library_manifest(
        block_manifest,
        schema="hytalerl_region_block_semantic_library_v1",
        expected_semantic=structure_row.get("block_semantic_library_sha256"),
        label=f"{structure} block-semantic library",
    )
    if block_manifest.get("source_region_library_semantic_sha256") != (
        region_manifest["library_semantic_sha256"]
    ):
        raise ValueError(f"block semantics name another Region library: {structure}")

    region_entries = {
        _sha256(value.get("semantic_sha256"), "Region entry semantic"): value
        for value in region_manifest.get("entries", [])
    }
    block_entries = {
        _sha256(
            value.get("source_region_semantic_sha256"),
            "block source Region semantic",
        ): value
        for value in block_manifest.get("entries", [])
    }
    artifacts: list[dict[str, Any]] = []
    for row in sorted(structure_row.get("artifacts", []), key=lambda item: item["seed"]):
        seed = _nonnegative_int(row.get("seed"), "artifact seed")
        identity = (structure, seed)
        analysis = analysis_rows.get(identity)
        if analysis is None:
            raise ValueError(f"analysis is missing {structure}/{seed}")
        split = _split(row.get("split"))
        if split_mapping.get(str(seed)) != split:
            raise ValueError(f"global seed split differs for {structure}/{seed}")
        region_semantic = _sha256(
            row.get("region_semantic_sha256"), "Region semantic"
        )
        region_entry = region_entries.get(region_semantic)
        block_entry = block_entries.get(region_semantic)
        if region_entry is None or block_entry is None:
            raise ValueError(f"library pairing missing for {structure}/{seed}")
        region_relative = _relative_path(row.get("region_file"))
        block_relative = _relative_path(row.get("block_semantic_file"))
        fluid = _mapping(row.get("fluid_semantics"), "fluid semantics")
        fluid_relative = _relative_path(fluid.get("file"))
        expected_region_path = _joined_relative(
            PurePosixPath(region_manifest_relative).parent,
            region_entry.get("path"),
        )
        expected_block_path = _joined_relative(
            PurePosixPath(block_manifest_relative).parent,
            block_entry.get("path"),
        )
        if region_relative != expected_region_path:
            raise ValueError(f"Region report/manifest path differs: {structure}/{seed}")
        if block_relative != expected_block_path:
            raise ValueError(f"block report/manifest path differs: {structure}/{seed}")
        region_file_sha = _sha256(row.get("region_file_sha256"), "Region file")
        block_file_sha = _sha256(
            row.get("block_semantic_file_sha256"), "block-semantic file"
        )
        fluid_file_sha = _sha256(fluid.get("file_sha256"), "fluid file")
        if region_file_sha != _sha256(region_entry.get("file_sha256"), "Region entry file"):
            raise ValueError(f"Region report/manifest hash differs: {structure}/{seed}")
        if block_file_sha != _sha256(block_entry.get("file_sha256"), "block entry file"):
            raise ValueError(f"block report/manifest hash differs: {structure}/{seed}")
        if verify_artifact_hashes:
            for relative, expected in (
                (region_relative, region_file_sha),
                (block_relative, block_file_sha),
                (fluid_relative, fluid_file_sha),
            ):
                _require_file_sha256(root, relative, expected)
        block_semantic = _sha256(
            row.get("block_semantic_sha256"), "block semantic"
        )
        fluid_semantic = _sha256(fluid.get("semantic_sha256"), "fluid semantic")
        if block_semantic != _sha256(
            block_entry.get("semantic_sha256"), "block entry semantic"
        ):
            raise ValueError(f"block semantic differs: {structure}/{seed}")
        if analysis.get("region_semantic_sha256") != region_semantic:
            raise ValueError(f"analysis Region differs: {structure}/{seed}")
        if analysis.get("block_semantic_sha256") != block_semantic:
            raise ValueError(f"analysis block semantics differ: {structure}/{seed}")
        if analysis.get("fluid_semantic_sha256") != fluid_semantic:
            raise ValueError(f"analysis fluid semantics differ: {structure}/{seed}")
        core = region_entry.get("core_min_chunk_xz")
        if not isinstance(core, list) or len(core) != 2:
            raise ValueError("Region entry core_min_chunk_xz changed")
        artifacts.append(
            {
                "structure": structure,
                "seed": seed,
                "split": split,
                "core_min_chunk_xz": [
                    _exact_int(core[0], "core X"),
                    _exact_int(core[1], "core Z"),
                ],
                "region": {
                    "path": region_relative,
                    "file_sha256": region_file_sha,
                    "semantic_sha256": region_semantic,
                },
                "block_semantics": {
                    "path": block_relative,
                    "file_sha256": block_file_sha,
                    "semantic_sha256": block_semantic,
                },
                "fluid_semantics": {
                    "path": fluid_relative,
                    "file_sha256": fluid_file_sha,
                    "semantic_sha256": fluid_semantic,
                },
                "spawn_position": _position(row.get("spawn_position")),
                "features": _compact_features(analysis),
            }
        )
    expected_count = _positive_int(
        structure_row.get("artifact_count"), f"{structure} artifact_count"
    )
    if len(artifacts) != expected_count:
        raise ValueError(f"structure artifact count differs: {structure}")
    return {
        "structure": structure,
        "artifact_count": expected_count,
        "region_library": {
            "manifest_path": region_manifest_relative,
            "manifest_file_sha256": _file_sha256(
                _artifact_path(root, region_manifest_relative)
            ),
            "library_semantic_sha256": region_manifest["library_semantic_sha256"],
        },
        "block_semantic_library": {
            "manifest_path": block_manifest_relative,
            "manifest_file_sha256": _file_sha256(
                _artifact_path(root, block_manifest_relative)
            ),
            "library_semantic_sha256": block_manifest["library_semantic_sha256"],
        },
        "traversal": {
            "available": False,
            "reason": "traversal_sidecars_not_compiled",
        },
        "artifacts": artifacts,
    }


def _compact_features(analysis: Mapping[str, Any]) -> dict[str, Any]:
    terrain = _mapping(analysis.get("terrain"), "terrain analysis")
    surface = _mapping(terrain.get("surface_height"), "surface height")
    step = _mapping(terrain.get("adjacent_surface_step"), "surface step")
    cave = _mapping(analysis.get("cave_overhang_proxy"), "cave proxy")
    materials = _mapping(analysis.get("materials"), "materials")
    fluid = _mapping(analysis.get("stable_fluid_identity"), "stable fluid")
    hazards = _mapping(analysis.get("fluids_and_hazards"), "hazards")
    material_assets = sorted(
        str(row["asset_id"])
        for row in materials.get("assets", [])
        if row.get("is_air") is False and row.get("asset_id") is not None
    )
    fluid_assets = sorted(
        str(row["asset_id"])
        for row in fluid.get("assets", [])
        if row.get("asset_id")
    )
    result = {
        "terrain_relief_blocks": _nonnegative_int(
            terrain.get("relief_blocks"), "terrain relief"
        ),
        "surface_height_minimum": _finite(surface.get("minimum"), "surface min"),
        "surface_height_median": _finite(surface.get("median"), "surface median"),
        "surface_height_maximum": _finite(surface.get("maximum"), "surface max"),
        "surface_step_maximum": _finite(step.get("maximum"), "step max"),
        "surface_step_mean": _finite(step.get("mean"), "step mean"),
        "surface_step_nonzero_fraction": _fraction(
            step.get("nonzero_fraction"), "step nonzero fraction"
        ),
        "subsurface_void_fraction": _fraction(
            cave.get("subsurface_void_fraction"), "subsurface void fraction"
        ),
        "unique_non_air_material_ids": _nonnegative_int(
            materials.get("unique_non_air_asset_ids"), "unique materials"
        ),
        "non_air_material_asset_ids": material_assets,
        "core_fluid_cells": _nonnegative_int(
            fluid.get("core_fluid_cells"), "core fluid cells"
        ),
        "full_fluid_cells": _nonnegative_int(
            fluid.get("full_fluid_cells"), "full fluid cells"
        ),
        "fluid_asset_ids": fluid_assets,
        "damaging_flag_cells": _nonnegative_int(
            hazards.get("damaging_flag_cells"), "damaging cells"
        ),
        "maximum_block_damage": _finite(
            hazards.get("maximum_block_damage"), "maximum block damage"
        ),
        "maximum_fluid_damage": _finite(
            hazards.get("maximum_fluid_damage"), "maximum fluid damage"
        ),
    }
    _exact_fields(result, _FEATURE_FIELDS, "compact environment features")
    return result


def _analysis_artifact_index(
    analysis: Mapping[str, Any],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for structure_row in analysis.get("structures", []):
        structure = _structure_id(structure_row.get("structure"))
        for row in structure_row.get("artifacts", []):
            seed = _nonnegative_int(row.get("seed"), "analysis seed")
            identity = (structure, seed)
            if identity in result:
                raise ValueError(f"duplicate analysis artifact: {identity}")
            result[identity] = row
    return result


def _require_report(report: Mapping[str, Any]) -> None:
    if report.get("schema") != "hytalerl_worldgen_v2_region_report_v1":
        raise ValueError("unsupported WorldGen V2 capture report")
    if _exact_int(report.get("version"), "capture report version") != 1:
        raise ValueError("unsupported WorldGen V2 capture report version")
    if report.get("world_template") != "hytale_generator":
        raise ValueError("capture report does not use the explicit V2 world")
    if report.get("worldgen_provider") != "HytaleGenerator":
        raise ValueError("capture report provider changed")
    semantic = _sha256(report.get("report_semantic_sha256"), "capture report")
    if semantic != _report_digest(report):
        raise ValueError("capture report semantic SHA-256 mismatch")
    required = (
        "block_asset_identity_sidecar",
        "block_semantic_round_trip",
        "calibration_and_heldout_split",
        "explicit_v2_world_template",
        "physical_palette_round_trip",
        "provider_and_structure_identity",
        "runtime_bridge_identity",
        "same_world_region_confirmation",
        "stable_fluid_forward_reverse_exact",
        "stable_fluid_identity_sidecar",
    )
    validated = _mapping(report.get("validated"), "capture validations")
    missing = [name for name in required if validated.get(name) is not True]
    if missing:
        raise ValueError(f"capture validations are incomplete: {missing}")
    if not isinstance(report.get("structures"), list) or not report["structures"]:
        raise ValueError("capture report has no structures")


def _require_analysis(
    analysis: Mapping[str, Any],
    report_path: Path,
    report: Mapping[str, Any],
) -> None:
    if analysis.get("schema") != "hytalerl_worldgen_v2_region_analysis_v2":
        raise ValueError("unsupported WorldGen V2 analysis report")
    if _exact_int(analysis.get("version"), "analysis version") != 2:
        raise ValueError("unsupported WorldGen V2 analysis version")
    if analysis.get("world_template") != report.get("world_template") or (
        analysis.get("worldgen_provider") != report.get("worldgen_provider")
    ):
        raise ValueError("analysis provider identity differs from capture")
    semantic = _sha256(analysis.get("report_semantic_sha256"), "analysis report")
    if semantic != _report_digest(analysis):
        raise ValueError("analysis report semantic SHA-256 mismatch")
    if _sha256(
        analysis.get("source_capture_report_file_sha256"),
        "analysis source report file",
    ) != _file_sha256(report_path):
        raise ValueError("analysis names a different capture report file")
    if _sha256(
        analysis.get("source_capture_report_semantic_sha256"),
        "analysis source report semantic",
    ) != report.get("report_semantic_sha256"):
        raise ValueError("analysis names a different capture report semantic")


def _require_plan(plan: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    if plan.get("schema") != "hytalerl_worldgen_v2_region_capture_plan_v1":
        raise ValueError("unsupported WorldGen V2 capture plan")
    if _exact_int(plan.get("version"), "capture plan version") != 1:
        raise ValueError("unsupported WorldGen V2 capture plan version")
    if plan.get("world_template") != report.get("world_template") or (
        plan.get("worldgen_provider") != report.get("worldgen_provider")
    ):
        raise ValueError("capture plan provider identity differs from report")
    seeds = plan.get("seeds")
    if not isinstance(seeds, list) or len(set(seeds)) != len(seeds):
        raise ValueError("capture plan seeds are invalid")
    if not isinstance(plan.get("structures"), list) or not plan["structures"]:
        raise ValueError("capture plan structures are invalid")
    if set(str(seed) for seed in seeds) != set(plan.get("seed_split", {})):
        raise ValueError("capture plan seed split is incomplete")


def _require_library_manifest(
    manifest: Mapping[str, Any],
    *,
    schema: str,
    expected_semantic: object,
    label: str,
) -> None:
    if manifest.get("schema") != schema:
        raise ValueError(f"{label} schema changed")
    if _exact_int(manifest.get("version"), f"{label} version") != 1:
        raise ValueError(f"{label} version changed")
    semantic = _sha256(manifest.get("library_semantic_sha256"), label)
    if semantic != _sha256(expected_semantic, f"expected {label}"):
        raise ValueError(f"{label} semantic differs from capture report")
    if schema == "hytalerl_region_block_semantic_library_v1":
        payload = {
            "schema": manifest.get("schema"),
            "version": manifest.get("version"),
            "source_region_library_semantic_sha256": manifest.get(
                "source_region_library_semantic_sha256"
            ),
            "block_semantic_contract_sha256": manifest.get(
                "block_semantic_contract_sha256"
            ),
            "physical_alignment_contract_sha256": manifest.get(
                "physical_alignment_contract_sha256"
            ),
            "entries": [
                {
                    "source_region_semantic_sha256": entry.get(
                        "source_region_semantic_sha256"
                    ),
                    "semantic_sha256": entry.get("semantic_sha256"),
                }
                for entry in manifest.get("entries", [])
            ],
        }
    else:
        payload = dict(manifest)
        payload.pop("library_semantic_sha256", None)
    if semantic != _canonical_sha256(payload):
        raise ValueError(f"{label} semantic SHA-256 mismatch")


def _parse_structure(
    value: object,
    *,
    split_contract: Mapping[str, str],
    seen_structures: set[str],
) -> list[V2JaxArtifact]:
    source = _mapping(value, "bundle structure")
    _exact_fields(source, _STRUCTURE_FIELDS, "bundle structure")
    structure = _structure_id(source["structure"])
    if structure in seen_structures:
        raise ValueError(f"duplicate bundle structure: {structure}")
    seen_structures.add(structure)
    for field in ("region_library", "block_semantic_library"):
        library = _mapping(source[field], field)
        _exact_fields(library, _LIBRARY_FIELDS, field)
        _relative_path(library["manifest_path"])
        _sha256(library["manifest_file_sha256"], f"{field} file")
        _sha256(library["library_semantic_sha256"], f"{field} semantic")
    traversal = _mapping(source["traversal"], "traversal")
    _exact_fields(traversal, _TRAVERSAL_FIELDS, "traversal")
    if traversal["available"] is not False or traversal["reason"] != (
        "traversal_sidecars_not_compiled"
    ):
        raise ValueError("bundle traversal availability changed")
    rows = source["artifacts"]
    count = _positive_int(source["artifact_count"], "structure artifact_count")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("structure artifacts differ from artifact_count")
    parsed: list[V2JaxArtifact] = []
    seeds: list[int] = []
    for row_value in rows:
        row = _mapping(row_value, "bundle artifact")
        _exact_fields(row, _ARTIFACT_FIELDS, "bundle artifact")
        if _structure_id(row["structure"]) != structure:
            raise ValueError("artifact structure differs from parent")
        seed = _nonnegative_int(row["seed"], "artifact seed")
        split = _split(row["split"])
        if split_contract.get(str(seed)) != split:
            raise ValueError("artifact split differs from global seed split")
        core = row["core_min_chunk_xz"]
        if not isinstance(core, list) or len(core) != 2:
            raise ValueError("artifact core must contain two integers")
        region = _validate_file_ref(row["region"], "Region artifact")
        block = _validate_file_ref(row["block_semantics"], "block semantics")
        fluid = _validate_file_ref(row["fluid_semantics"], "fluid semantics")
        features = _mapping(row["features"], "environment features")
        _exact_fields(features, _FEATURE_FIELDS, "environment features")
        parsed.append(
            V2JaxArtifact(
                structure=structure,
                seed=seed,
                split=split,
                core_min_chunk_xz=(
                    _exact_int(core[0], "core X"),
                    _exact_int(core[1], "core Z"),
                ),
                region_path=region["path"],
                region_file_sha256=region["file_sha256"],
                region_semantic_sha256=region["semantic_sha256"],
                block_semantic_path=block["path"],
                block_semantic_file_sha256=block["file_sha256"],
                block_semantic_sha256=block["semantic_sha256"],
                fluid_semantic_path=fluid["path"],
                fluid_semantic_file_sha256=fluid["file_sha256"],
                fluid_semantic_sha256=fluid["semantic_sha256"],
                spawn_position=tuple(_position(row["spawn_position"])),
                features=_json_copy(features),
            )
        )
        seeds.append(seed)
    if seeds != sorted(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("structure artifacts must use unique seed order")
    return parsed


def _validate_split_contract(value: object) -> dict[str, str]:
    source = _mapping(value, "measurement split contract")
    _exact_fields(source, _SPLIT_FIELDS, "measurement split contract")
    if source["method"] != "sha256_seed_rank_v1":
        raise ValueError("unsupported V2 measurement split method")
    _sha256(source["salt_sha256"], "measurement split salt")
    if source["scope"] != "seed_identity_shared_across_all_structures":
        raise ValueError("V2 measurement split scope changed")
    mapping = _mapping(source["mapping"], "measurement split mapping")
    result: dict[str, str] = {}
    for raw_seed, raw_split in mapping.items():
        seed = _nonnegative_int(int(raw_seed), "split seed")
        if str(seed) != str(raw_seed):
            raise ValueError("measurement split seed key is not canonical")
        result[str(seed)] = _split(raw_split)
    if not result or set(result.values()) != _SPLITS:
        raise ValueError("measurement split requires calibration and heldout")
    return result


def _validate_capabilities(value: object) -> frozenset[str]:
    source = _mapping(value, "capabilities")
    _exact_fields(source, frozenset(_CAPABILITIES), "capabilities")
    available: set[str] = set()
    for name in _CAPABILITIES:
        row = _mapping(source[name], f"capability {name}")
        _exact_fields(row, _CAPABILITY_FIELDS, f"capability {name}")
        if not isinstance(row["available"], bool):
            raise TypeError(f"capability {name} availability must be bool")
        if row["reason"] != _CAPABILITY_REASON[name]:
            raise ValueError(f"capability {name} reason changed")
        if row["available"]:
            available.add(name)
    return frozenset(available)


def _capability_manifest() -> dict[str, dict[str, object]]:
    return {
        name: {
            "available": name in _AVAILABLE_CAPABILITIES,
            "reason": _CAPABILITY_REASON[name],
        }
        for name in _CAPABILITIES
    }


def _verify_source_files(root: Path, source: Mapping[str, Any]) -> None:
    report_ref = _validate_file_ref(source["capture_report"], "capture report")
    analysis_ref = _validate_file_ref(source["analysis_report"], "analysis report")
    plan_ref = _validate_file_ref(source["capture_plan"], "capture plan")
    for reference in (report_ref, analysis_ref, plan_ref):
        _require_file_sha256(root, reference["path"], reference["file_sha256"])
    report = _json_object(_artifact_path(root, report_ref["path"]))
    analysis = _json_object(_artifact_path(root, analysis_ref["path"]))
    plan = _json_object(_artifact_path(root, plan_ref["path"]))
    if _report_digest(report) != report_ref["semantic_sha256"]:
        raise ValueError("source capture report semantic changed")
    if _report_digest(analysis) != analysis_ref["semantic_sha256"]:
        raise ValueError("source analysis report semantic changed")
    if _canonical_sha256(plan) != plan_ref["semantic_sha256"]:
        raise ValueError("source capture plan semantic changed")


def _validate_file_ref(value: object, label: str) -> dict[str, str]:
    source = _mapping(value, label)
    _exact_fields(source, _FILE_REF_FIELDS, label)
    return {
        "path": _relative_path(source["path"]),
        "file_sha256": _sha256(source["file_sha256"], f"{label} file"),
        "semantic_sha256": _sha256(
            source["semantic_sha256"], f"{label} semantic"
        ),
    }


def _file_ref(
    root: Path,
    relative: str,
    *,
    semantic_sha256: object,
) -> dict[str, str]:
    safe = _relative_path(relative)
    return {
        "path": safe,
        "file_sha256": _file_sha256(_artifact_path(root, safe)),
        "semantic_sha256": _sha256(semantic_sha256, "file reference semantic"),
    }


def _require_file_sha256(root: Path, relative: str, expected: str) -> None:
    path = _artifact_path(root, relative)
    if not path.is_file():
        raise ValueError(f"bundle artifact is missing: {relative}")
    if _file_sha256(path) != _sha256(expected, "expected file"):
        raise ValueError(f"bundle artifact file SHA-256 mismatch: {relative}")


def _report_digest(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("report_semantic_sha256", None)
    return _canonical_sha256(payload)


def _bundle_digest(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("bundle_semantic_sha256", None)
    return _canonical_sha256(payload)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_path(root: Path, relative: object) -> Path:
    safe = _relative_path(relative)
    path = (root / PurePosixPath(safe)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("bundle artifact path escapes its root")
    return path


def _joined_relative(parent: PurePosixPath, child: object) -> str:
    return _relative_path((parent / _relative_path(child)).as_posix())


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("bundle paths must be non-empty POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("bundle paths must remain under the artifact root")
    return path.as_posix()


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
        raise ValueError(f"JSON document must be an object: {path}")
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
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _bundle_id(value: object) -> str:
    if not isinstance(value, str) or _SAFE_BUNDLE_ID.fullmatch(value) is None:
        raise ValueError("bundle_id must be one safe path component")
    return value


def _structure_id(value: object) -> str:
    text = _nonempty(value, "structure")
    if "/" in text or "\\" in text or text in {".", ".."}:
        raise ValueError("structure must not contain path separators")
    return text


def _position(value: object) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("spawn_position must contain x, y, and z")
    return [_finite(item, "spawn coordinate") for item in value]


def _split(value: object) -> str:
    if value not in _SPLITS:
        raise ValueError("split must be calibration or heldout")
    return str(value)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256")
    return value.lower()


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _exact_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error


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


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _fraction(value: object, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


__all__ = [
    "BUNDLE_ASSIGNMENT_METHOD",
    "BUNDLE_AUTHORITY",
    "BUNDLE_RUNTIME_SOURCE",
    "BUNDLE_SCHEMA",
    "BUNDLE_VERSION",
    "CAPABILITY_BIOME_IDENTITY",
    "CAPABILITY_DYNAMIC_ENTITIES",
    "CAPABILITY_EXACT_GEOMETRY",
    "CAPABILITY_NATIVE_LIGHTING",
    "CAPABILITY_NATIVE_TRAVERSAL",
    "CAPABILITY_STABLE_BLOCKS",
    "CAPABILITY_STABLE_FLUIDS",
    "CAPABILITY_STATIC_SPAWN",
    "CAPABILITY_STRUCTURE_INSTANCES",
    "V2JaxArtifact",
    "V2JaxBundle",
    "compile_capture_bundle",
]
