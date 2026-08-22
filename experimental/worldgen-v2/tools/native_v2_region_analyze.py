"""Measure bounded WorldGen V2 Region evidence without inventing labels.

The capture contract exposes exact physical cells, block-asset identities,
spawn positions, fluids, and hazards.  It does not expose per-cell biome IDs,
road markers, or authored-structure instances.  This analyzer measures the
former and records the latter as unobservable instead of deriving them from
terrain shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from _workspace import use_workspace_hytalerl


use_workspace_hytalerl()

from hytalegym.geometry.contract import (  # noqa: E402
    FLAG_DAMAGING,
    FLAG_FLUID,
    FLAG_SOLID,
)
from hytalegym.worldgen.region import (  # noqa: E402
    CAPTURE_CHUNKS_PER_AXIS,
    CAPTURE_HALO_CHUNKS,
    CHUNK_SIZE,
    CORE_BLOCKS_PER_AXIS,
    HEIGHT_SECTIONS,
    NativeRegionBlockSemanticSnapshot,
    NativeRegionSnapshot,
    RegionArtifactLibrary,
    RegionBlockSemanticLibrary,
    load_region_library_diversity_report,
)
from hytalegym.worldgen.region.block_action_catalog import (  # noqa: E402
    resolve_region_block_asset_references,
)
from hytalegym.worldgen.surrogate import HytaleAssetArchive  # noqa: E402

from native_v2_region_fluid_sidecar import (  # noqa: E402
    NativeRegionFluidSemanticSnapshot,
    require_fluid_snapshot_identity,
)


CAPTURE_REPORT_SCHEMA = "hytalerl_worldgen_v2_region_report_v1"
ANALYSIS_SCHEMA = "hytalerl_worldgen_v2_region_analysis_v2"
ANALYSIS_VERSION = 2
WORLD_TEMPLATE = "hytale_generator"
WORLDGEN_PROVIDER = "HytaleGenerator"


def dense_region(values: np.ndarray) -> np.ndarray:
    """Decode section-major cells into ``[x, y, z]`` capture coordinates."""

    source = np.asarray(values)
    expected = (
        CAPTURE_CHUNKS_PER_AXIS**2,
        HEIGHT_SECTIONS,
        CHUNK_SIZE**3,
    )
    if source.shape != expected:
        raise ValueError(
            f"Region section payload has shape {source.shape}, expected {expected}"
        )
    return source.reshape(
        CAPTURE_CHUNKS_PER_AXIS,
        CAPTURE_CHUNKS_PER_AXIS,
        HEIGHT_SECTIONS,
        CHUNK_SIZE,
        CHUNK_SIZE,
        CHUNK_SIZE,
    ).transpose(0, 5, 2, 3, 1, 4).reshape(
        CAPTURE_CHUNKS_PER_AXIS * CHUNK_SIZE,
        HEIGHT_SECTIONS * CHUNK_SIZE,
        CAPTURE_CHUNKS_PER_AXIS * CHUNK_SIZE,
    )


def region_distribution_metrics(
    snapshot: NativeRegionSnapshot,
    sidecar: NativeRegionBlockSemanticSnapshot,
    *,
    fluid_sidecar: NativeRegionFluidSemanticSnapshot | None = None,
    asset_references: Mapping[tuple[int, ...], str] | None = None,
) -> dict[str, Any]:
    """Measure one exact Region core and retain explicit proxy boundaries."""

    if not isinstance(snapshot, NativeRegionSnapshot):
        raise TypeError("snapshot must be NativeRegionSnapshot")
    if not isinstance(sidecar, NativeRegionBlockSemanticSnapshot):
        raise TypeError("sidecar must be NativeRegionBlockSemanticSnapshot")
    if sidecar.source_region_semantic_sha256 != snapshot.semantic_digest():
        raise ValueError("block-semantic sidecar names another Region")
    if not np.array_equal(sidecar.core_min_chunk_xz, snapshot.core_min_chunk_xz):
        raise ValueError("block-semantic sidecar core differs from Region")

    physical = dense_region(snapshot.cell_code)
    semantic = dense_region(sidecar.cell_code)
    start = CAPTURE_HALO_CHUNKS * CHUNK_SIZE
    stop = start + CORE_BLOCKS_PER_AXIS
    physical = physical[start:stop, :, start:stop]
    semantic = semantic[start:stop, :, start:stop]

    palette = snapshot.cell_palette
    flags = palette.flags[physical].astype(np.uint16)
    solid = (flags & np.uint16(FLAG_SOLID)) != 0
    fluid = (flags & np.uint16(FLAG_FLUID)) != 0
    damaging = (flags & np.uint16(FLAG_DAMAGING)) != 0

    surface, active = _surface_height_grid(solid)
    active_surface = surface[active]
    if not active_surface.size:
        raise ValueError("Region core contains no solid surface columns")
    slope = _surface_slope_metrics(surface, active)

    top_cell = np.where(active, surface - 1, -1)
    y = np.arange(solid.shape[1], dtype=np.int16)[None, :, None]
    below_surface = active[:, None, :] & (y < top_cell[:, None, :])
    subsurface_void = below_surface & ~solid
    void_by_column = np.count_nonzero(subsurface_void, axis=1)
    subsurface_cells = int(np.count_nonzero(below_surface))
    cave_cells = int(np.count_nonzero(subsurface_void))

    solid_below = solid & below_surface
    solid_starts = solid_below.copy()
    solid_starts[:, 1:, :] &= ~solid_below[:, :-1, :]
    solid_run_count = np.count_nonzero(solid_starts, axis=1)
    enclosed_void_start = subsurface_void.copy()
    enclosed_void_start[:, 0, :] = False
    enclosed_void_start[:, 1:, :] &= solid[:, :-1, :]
    enclosed_void_runs = np.count_nonzero(enclosed_void_start, axis=1)

    material = _material_metrics(
        semantic,
        sidecar,
        fluid=fluid,
        stable_fluid_measured=fluid_sidecar is not None,
        asset_references=asset_references,
    )
    hazard = _hazard_metrics(physical, palette, fluid, damaging)
    stable_fluid = _stable_fluid_metrics(
        snapshot,
        fluid_sidecar,
        physical_core_fluid=fluid,
    )
    spawn = _spawn_metrics(snapshot, surface, active)

    return {
        "seed": _nonnegative_int(snapshot.metadata.get("seed"), "seed"),
        "structure": _nonempty_string(
            snapshot.metadata.get("worldgen_structure"),
            "worldgen_structure",
        ),
        "region_semantic_sha256": snapshot.semantic_digest(),
        "block_semantic_sha256": sidecar.semantic_sha256(),
        "fluid_semantic_sha256": (
            fluid_sidecar.semantic_sha256()
            if fluid_sidecar is not None
            else None
        ),
        "scope": {
            "capture_blocks_xz": CAPTURE_CHUNKS_PER_AXIS * CHUNK_SIZE,
            "capture_height_blocks": HEIGHT_SECTIONS * CHUNK_SIZE,
            "measured_core_blocks_xz": CORE_BLOCKS_PER_AXIS,
            "measured_core_columns": CORE_BLOCKS_PER_AXIS**2,
            "measured_core_cells": int(physical.size),
        },
        "terrain": {
            "surface_definition": "highest_FLAG_SOLID_cell_ceiling_v1",
            "warning": (
                "Surface height is a block-cell ceiling and does not apply "
                "fractional collision-shape tops."
            ),
            "active_surface_columns": int(np.count_nonzero(active)),
            "empty_surface_columns": int(np.count_nonzero(~active)),
            "surface_height": _numeric_summary(active_surface),
            "relief_blocks": int(np.max(active_surface) - np.min(active_surface)),
            "adjacent_surface_step": slope,
        },
        "cave_overhang_proxy": {
            "definition": (
                "non-solid cells below each column's highest solid cell; "
                "this is a column proxy, not connected-component cave identity"
            ),
            "subsurface_cells": subsurface_cells,
            "subsurface_void_cells": cave_cells,
            "subsurface_void_fraction": (
                float(cave_cells / subsurface_cells)
                if subsurface_cells
                else 0.0
            ),
            "columns_with_subsurface_void": int(np.count_nonzero(void_by_column)),
            "columns_with_subsurface_void_fraction": float(
                np.count_nonzero(void_by_column) / np.count_nonzero(active)
            ),
            "maximum_subsurface_void_cells_per_column": int(
                np.max(void_by_column, initial=0)
            ),
            "enclosed_void_run_count": int(np.sum(enclosed_void_runs)),
            "columns_with_multiple_solid_runs": int(
                np.count_nonzero(solid_run_count > 1)
            ),
        },
        "materials": material,
        "fluids_and_hazards": hazard,
        "stable_fluid_identity": stable_fluid,
        "spawn": spawn,
    }


def analyze_capture_report(
    capture_report: str | Path,
    *,
    output: str | Path | None = None,
    expected_capture_report_sha256: str | None = None,
    measurement_split_salt_sha256: str | None = None,
    resolve_block_asset_ids: bool = False,
) -> dict[str, Any]:
    """Reload all capture artifacts, analyze them, and write one exact report."""

    source = Path(capture_report).resolve()
    report = _json_object(source)
    if report.get("schema") != CAPTURE_REPORT_SCHEMA:
        raise ValueError("unsupported WorldGen V2 Region capture report")
    if report.get("world_template") != WORLD_TEMPLATE:
        raise ValueError("analysis accepts only the explicit V2 world template")
    if report.get("worldgen_provider") != WORLDGEN_PROVIDER:
        raise ValueError("analysis requires the HytaleGenerator provider")
    if _report_sha256(report) != report.get("report_semantic_sha256"):
        raise ValueError("capture report semantic SHA-256 mismatch")
    source_file_sha256 = _file_sha256(source)
    if expected_capture_report_sha256 is not None and source_file_sha256 != (
        _sha256(expected_capture_report_sha256, "expected capture report")
    ):
        raise ValueError("capture report file SHA-256 differs from expectation")

    root = source.parent
    plan_path = root / _relative_path(report.get("capture_plan"))
    plan = _json_object(plan_path)
    if _canonical_sha256(plan) != report.get("capture_plan_sha256"):
        raise ValueError("capture plan semantic SHA-256 mismatch")
    split_contract = _measurement_split_contract(
        report,
        plan,
        override_salt_sha256=measurement_split_salt_sha256,
    )
    provenance = _mapping(report.get("provenance"), "provenance")
    bridge_provenance = _mapping(
        provenance.get("bridge_jar"),
        "bridge provenance",
    )
    expected_bridge = _sha256(
        bridge_provenance.get("sha256"),
        "capture bridge",
    )
    capture_has_fluid_semantics = bool(
        _mapping(plan.get("capture_contract"), "capture contract").get(
            "stable_fluid_identity"
        )
    )
    asset_archive: HytaleAssetArchive | None = None
    asset_resolution: dict[str, Any]
    if resolve_block_asset_ids:
        assets_provenance = _mapping(provenance.get("assets"), "assets provenance")
        assets_path = Path(
            _nonempty_string(assets_provenance.get("path"), "assets path")
        ).resolve()
        expected_assets = _sha256(
            assets_provenance.get("sha256"), "capture assets"
        )
        if _file_sha256(assets_path) != expected_assets:
            raise ValueError("installed assets differ from capture provenance")
        asset_archive = HytaleAssetArchive(assets_path)
        asset_resolution = {
            "status": "resolved_from_capture_assets",
            "assets_sha256": expected_assets,
            "identity": "reverse_sha256_asset_key_against_installed_assets",
            "scope": "block_assets_only_not_fluid_assets",
        }
    else:
        asset_resolution = {
            "status": "not_requested",
            "scope": "stable_block_asset_hashes_only",
        }
    structures = []
    raw_structures = report.get("structures")
    if not isinstance(raw_structures, list) or not raw_structures:
        raise ValueError("capture report contains no structures")
    try:
        for raw_structure in raw_structures:
            structure = _mapping(raw_structure, "structure report")
            library = RegionArtifactLibrary.load(
                root / _relative_path(structure.get("region_library_manifest"))
            )
            semantic_library = RegionBlockSemanticLibrary.load(
                root
                / _relative_path(
                    structure.get("block_semantic_library_manifest")
                ),
                library,
            )
            diversity = load_region_library_diversity_report(
                root / _relative_path(
                    structure.get("geometric_diversity_report")
                ),
                library=library,
            )
            raw_artifacts = structure.get("artifacts")
            if not isinstance(raw_artifacts, list):
                raise ValueError("capture structure artifacts must be a list")
            source_rows = {
                _sha256(row["region_semantic_sha256"], "Region row semantic"):
                row
                for row in raw_artifacts
                if isinstance(row, dict)
            }
            if len(source_rows) != len(library.entries):
                raise ValueError("capture report rows differ from Region library")
            sidecars = semantic_library.load_for_region_entries(library.entries)
            asset_references: dict[tuple[int, ...], str] | None = None
            if asset_archive is not None:
                resolved = resolve_region_block_asset_references(
                    asset_archive,
                    tuple(
                        palette_entry
                        for sidecar in sidecars
                        for palette_entry in sidecar.palette
                    ),
                )
                asset_references = {
                    entry.asset_key: reference
                    for entry, reference in resolved
                }
            artifact_metrics = []
            surfaces = []
            for entry, sidecar in zip(library.entries, sidecars, strict=True):
                snapshot_path = library.manifest_path.parent / entry.path
                snapshot = NativeRegionSnapshot.load(snapshot_path)
                source_row = source_rows.get(entry.semantic_sha256)
                if source_row is None:
                    raise ValueError(
                        "Region library entry is absent from capture report"
                    )
                seed = str(entry.seed)
                source_split = _nonempty_string(
                    source_row.get("split"), "source split"
                )
                if source_split != split_contract["source_mapping"].get(seed):
                    raise ValueError(
                        "capture artifact split differs from source mapping"
                    )
                metrics = region_distribution_metrics(
                    snapshot,
                    sidecar,
                    fluid_sidecar=_load_fluid_sidecar(
                        root,
                        source_row,
                        snapshot,
                        expected_bridge=expected_bridge,
                        required=capture_has_fluid_semantics,
                    ),
                    asset_references=asset_references,
                )
                metrics["split"] = split_contract["effective_mapping"][seed]
                metrics["source_report_split"] = source_split
                artifact_metrics.append(metrics)
                surfaces.append(
                    (
                        metrics["seed"],
                        *_surface_height_grid(_core_solid_grid(snapshot)),
                    )
                )

            structures.append(
                {
                    "structure": _nonempty_string(
                        structure.get("structure"), "structure"
                    ),
                    "artifact_count": len(artifact_metrics),
                    "region_library_semantic_sha256": library.semantic_sha256,
                    "block_semantic_library_sha256": (
                        semantic_library.semantic_sha256
                    ),
                    "source_diversity_report_semantic_sha256": diversity[
                        "report_semantic_sha256"
                    ],
                    "pairwise_surface_comparison": _pairwise_surface_metrics(
                        surfaces
                    ),
                    "material_asset_union_including_air": sorted(
                        {
                            row["asset_key_sha256"]
                            for artifact in artifact_metrics
                            for row in artifact["materials"]["assets"]
                        }
                    ),
                    "material_asset_id_union_non_air": sorted(
                        {
                            row["asset_id"]
                            for artifact in artifact_metrics
                            for row in artifact["materials"]["assets"]
                            if row["asset_id"] is not None
                        }
                    ),
                    "stable_fluid_asset_union": sorted(
                        {
                            row["asset_id"]
                            for artifact in artifact_metrics
                            for row in artifact["stable_fluid_identity"].get(
                                "assets", []
                            )
                        }
                    ),
                    "artifacts": sorted(
                        artifact_metrics,
                        key=lambda value: int(value["seed"]),
                    ),
                }
            )
    finally:
        if asset_archive is not None:
            asset_archive.close()

    analysis: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "version": ANALYSIS_VERSION,
        "source_capture_report": source.name,
        "source_capture_report_file_sha256": source_file_sha256,
        "source_capture_report_semantic_sha256": report[
            "report_semantic_sha256"
        ],
        "world_template": WORLD_TEMPLATE,
        "worldgen_provider": WORLDGEN_PROVIDER,
        "measurement_split_contract": split_contract,
        "block_asset_resolution": asset_resolution,
        "observability": {
            "measured": [
                "block-cell surface height and adjacent relief",
                "column cave/overhang proxy",
                "stable block-asset occupancy",
                "fluid and hazard occupancy",
                *(
                    ["stable Fluid.getId() identity and occupancy"]
                    if capture_has_fluid_semantics
                    else []
                ),
                "spawn placement relative to the measured surface",
            ],
            "unmeasured": {
                "biome_distribution": (
                    "capture protocol exposes no per-cell biome identity"
                ),
                "road_distribution": (
                    "capture protocol exposes no road instance or marker identity"
                ),
                "structure_distribution": (
                    "selected worldgen_structure names the generator preset, "
                    "not placed authored-structure instances"
                ),
                **(
                    {}
                    if capture_has_fluid_semantics
                    else {
                        "fluid_asset_identity": (
                            "capture has no stable fluid-semantic sidecar"
                        )
                    }
                ),
                "connected_cave_identity": (
                    "requires a 3D connected-component analyzer beyond the "
                    "published column proxy"
                ),
            },
        },
        "structures": sorted(
            structures,
            key=lambda value: str(value["structure"]),
        ),
        "report_semantic_sha256": "",
    }
    analysis["report_semantic_sha256"] = _report_sha256(analysis)
    destination = (
        Path(output).resolve()
        if output is not None
        else source.with_name("analysis.json")
    )
    if destination == source:
        raise ValueError("analysis output cannot overwrite its capture report")
    _atomic_json(destination, analysis)
    return analysis


def _core_solid_grid(snapshot: NativeRegionSnapshot) -> np.ndarray:
    dense = dense_region(snapshot.cell_code)
    start = CAPTURE_HALO_CHUNKS * CHUNK_SIZE
    stop = start + CORE_BLOCKS_PER_AXIS
    codes = dense[start:stop, :, start:stop]
    return (
        snapshot.cell_palette.flags[codes].astype(np.uint16)
        & np.uint16(FLAG_SOLID)
    ) != 0


def _surface_height_grid(solid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(solid, dtype=np.bool_)
    if values.ndim != 3:
        raise ValueError("solid grid must have shape [x, y, z]")
    active = np.any(values, axis=1)
    top = values.shape[1] - 1 - np.argmax(values[:, ::-1, :], axis=1)
    surface = np.where(active, top + 1, -1).astype(np.int16)
    return surface, active


def _surface_slope_metrics(
    surface: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    values = []
    for axis in (0, 1):
        left = [slice(None), slice(None)]
        right = [slice(None), slice(None)]
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        pair = active[tuple(left)] & active[tuple(right)]
        delta = np.abs(
            surface[tuple(right)].astype(np.int32)
            - surface[tuple(left)].astype(np.int32)
        )
        values.append(delta[pair])
    combined = np.concatenate(values) if values else np.empty(0, dtype=np.int32)
    summary = _numeric_summary(combined)
    summary["nonzero_fraction"] = (
        float(np.count_nonzero(combined) / combined.size)
        if combined.size
        else 0.0
    )
    return summary


def _material_metrics(
    semantic: np.ndarray,
    sidecar: NativeRegionBlockSemanticSnapshot,
    *,
    fluid: np.ndarray,
    stable_fluid_measured: bool,
    asset_references: Mapping[tuple[int, ...], str] | None,
) -> dict[str, Any]:
    counts = np.bincount(
        semantic.reshape(-1),
        minlength=len(sidecar.palette),
    ).astype(np.int64)
    by_asset: dict[str, int] = {}
    fluid_by_asset: dict[str, int] = {}
    asset_ids: dict[str, str | None] = {}
    air_key = None
    for code in np.flatnonzero(counts):
        entry = sidecar.palette[int(code)]
        asset_key = _asset_key_hex(entry.asset_key)
        by_asset[asset_key] = by_asset.get(asset_key, 0) + int(counts[code])
        reference = (
            asset_references.get(entry.asset_key)
            if asset_references is not None
            else None
        )
        previous = asset_ids.setdefault(asset_key, reference)
        if previous != reference:
            raise ValueError("one block asset key resolved to multiple IDs")
        if not any(entry.asset_key):
            air_key = asset_key
    fluid_counts = np.bincount(
        semantic[np.asarray(fluid, dtype=np.bool_)],
        minlength=len(sidecar.palette),
    ).astype(np.int64)
    for code in np.flatnonzero(fluid_counts):
        asset_key = _asset_key_hex(sidecar.palette[int(code)].asset_key)
        fluid_by_asset[asset_key] = (
            fluid_by_asset.get(asset_key, 0) + int(fluid_counts[code])
        )
    total = int(semantic.size)
    air_cells = int(by_asset.get(air_key, 0)) if air_key is not None else 0
    non_air = {key: count for key, count in by_asset.items() if key != air_key}
    non_air_total = sum(non_air.values())
    probabilities = (
        np.asarray(tuple(non_air.values()), dtype=np.float64) / non_air_total
        if non_air_total
        else np.empty(0, dtype=np.float64)
    )
    entropy = float(-np.sum(probabilities * np.log2(probabilities))) if (
        probabilities.size
    ) else 0.0
    return {
        "identity": "sha256_hytale_block_asset_key",
        "unique_asset_ids_including_air": len(by_asset),
        "unique_non_air_asset_ids": len(non_air),
        "air_cells": air_cells,
        "non_air_cells": non_air_total,
        "non_air_asset_entropy_bits": entropy,
        "effective_non_air_asset_count": float(2.0**entropy),
        "dominant_non_air_asset_fraction": (
            float(max(non_air.values()) / non_air_total)
            if non_air_total
            else 0.0
        ),
        "fluid_identity_boundary": {
            "status": (
                "measured_in_separate_stable_fluid_identity"
                if stable_fluid_measured
                else "unmeasured_no_fluid_asset_key_in_capture_contract"
            ),
            "fluid_cells_over_air_block": int(fluid_by_asset.get(air_key, 0))
            if air_key is not None
            else 0,
            "fluid_cells_over_non_air_block": int(
                sum(
                    count
                    for key, count in fluid_by_asset.items()
                    if key != air_key
                )
            ),
        },
        "assets": [
            {
                "asset_key_sha256": key,
                "asset_id": asset_ids.get(key),
                "is_air": key == air_key,
                "cell_count": count,
                "cell_fraction": float(count / total),
                "fluid_cell_overlap": int(fluid_by_asset.get(key, 0)),
            }
            for key, count in sorted(
                by_asset.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
    }


def _load_fluid_sidecar(
    root: Path,
    source_row: Mapping[str, Any],
    snapshot: NativeRegionSnapshot,
    *,
    expected_bridge: str,
    required: bool,
) -> NativeRegionFluidSemanticSnapshot | None:
    raw = source_row.get("fluid_semantics")
    if raw is None:
        if required:
            raise ValueError("capture contract requires a fluid-semantic sidecar")
        return None
    fluid = _mapping(raw, "fluid semantics row")
    path = root / _relative_path(fluid.get("file"))
    expected_file = _sha256(fluid.get("file_sha256"), "fluid sidecar file")
    if _file_sha256(path) != expected_file:
        raise ValueError("fluid-semantic sidecar file SHA-256 differs")
    sidecar = NativeRegionFluidSemanticSnapshot.load(path)
    expected_semantic = _sha256(
        fluid.get("semantic_sha256"),
        "fluid sidecar semantic",
    )
    if sidecar.semantic_sha256() != expected_semantic:
        raise ValueError("fluid-semantic sidecar semantic SHA-256 differs")
    require_fluid_snapshot_identity(
        sidecar,
        snapshot,
        evidence_bridge_sha256=expected_bridge,
    )
    expected_full = _string_int_mapping(
        fluid.get("full_asset_cell_counts"),
        "full fluid asset counts",
    )
    expected_core = _string_int_mapping(
        fluid.get("core_asset_cell_counts"),
        "core fluid asset counts",
    )
    if sidecar.asset_cell_counts() != expected_full:
        raise ValueError("full fluid-semantic asset counts differ")
    if sidecar.asset_cell_counts(core_only=True) != expected_core:
        raise ValueError("core fluid-semantic asset counts differ")
    return sidecar


def _stable_fluid_metrics(
    snapshot: NativeRegionSnapshot,
    sidecar: NativeRegionFluidSemanticSnapshot | None,
    *,
    physical_core_fluid: np.ndarray,
) -> dict[str, Any]:
    if sidecar is None:
        return {
            "status": "unmeasured_no_fluid_semantic_sidecar",
            "identity": None,
            "assets": [],
        }
    fluid_dense = dense_region(sidecar.cell_code)
    start = CAPTURE_HALO_CHUNKS * CHUNK_SIZE
    stop = start + CORE_BLOCKS_PER_AXIS
    fluid_core = fluid_dense[start:stop, :, start:stop]
    if not np.array_equal(fluid_core != 0, physical_core_fluid):
        raise ValueError("core stable-fluid identity differs from FLAG_FLUID")
    core_counts = sidecar.asset_cell_counts(core_only=True)
    full_counts = sidecar.asset_cell_counts()
    return {
        "status": "measured_exact",
        "identity": "stable_hytale_fluid_getId_string",
        "source_region_semantic_sha256": (
            sidecar.source_region_semantic_sha256
        ),
        "semantic_sha256": sidecar.semantic_sha256(),
        "full_fluid_cells": sum(full_counts.values()),
        "core_fluid_cells": sum(core_counts.values()),
        "unique_asset_ids_full": len(full_counts),
        "unique_asset_ids_core": len(core_counts),
        "assets": [
            {
                "asset_id": asset_id,
                "full_cell_count": full_counts[asset_id],
                "core_cell_count": core_counts.get(asset_id, 0),
                "full_cell_fraction": float(
                    full_counts[asset_id] / snapshot.cell_code.size
                ),
                "core_cell_fraction": float(
                    core_counts.get(asset_id, 0) / physical_core_fluid.size
                ),
            }
            for asset_id in sorted(full_counts)
        ],
    }


def _hazard_metrics(
    physical: np.ndarray,
    palette: Any,
    fluid: np.ndarray,
    damaging: np.ndarray,
) -> dict[str, Any]:
    counts = np.bincount(
        physical.reshape(-1),
        minlength=palette.size,
    ).astype(np.int64)
    used = counts > 0
    block_damage = palette.block_damage != 0
    fluid_damage = palette.fluid_damage != 0
    total = int(physical.size)
    return {
        "fluid_cells": int(np.count_nonzero(fluid)),
        "fluid_fraction": float(np.count_nonzero(fluid) / total),
        "damaging_flag_cells": int(np.count_nonzero(damaging)),
        "damaging_flag_fraction": float(np.count_nonzero(damaging) / total),
        "nonzero_block_damage_cells": int(np.sum(counts[block_damage])),
        "nonzero_fluid_damage_cells": int(np.sum(counts[fluid_damage])),
        "maximum_block_damage": int(
            np.max(palette.block_damage[used], initial=0)
        ),
        "maximum_fluid_damage": int(
            np.max(palette.fluid_damage[used], initial=0)
        ),
        "maximum_fluid_level": int(
            np.max(palette.fluid_level[used], initial=0)
        ),
        "maximum_fluid_fill_height": float(
            np.max(palette.fluid_fill_height[used], initial=0.0)
        ),
    }


def _spawn_metrics(
    snapshot: NativeRegionSnapshot,
    surface: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    position = np.asarray(snapshot.metadata.get("spawn_position"), dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("Region spawn_position must be finite xyz")
    core_origin = snapshot.core_min_chunk_xz.astype(np.int64) * CHUNK_SIZE
    local_x = int(math.floor(position[0])) - int(core_origin[0])
    local_z = int(math.floor(position[2])) - int(core_origin[1])
    inside = (
        0 <= local_x < CORE_BLOCKS_PER_AXIS
        and 0 <= local_z < CORE_BLOCKS_PER_AXIS
    )
    surface_height: int | None = None
    delta: float | None = None
    if inside and active[local_x, local_z]:
        surface_height = int(surface[local_x, local_z])
        delta = float(position[1] - surface_height)
    return {
        "position": [float(value) for value in position],
        "inside_measured_core": inside,
        "local_core_column_xz": [local_x, local_z] if inside else None,
        "surface_cell_ceiling": surface_height,
        "feet_y_minus_surface_cell_ceiling": delta,
    }


def _pairwise_surface_metrics(
    surfaces: Sequence[tuple[int, np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    result = []
    ordered = sorted(surfaces, key=lambda value: int(value[0]))
    for index, (left_seed, left, left_active) in enumerate(ordered):
        for right_seed, right, right_active in ordered[index + 1 :]:
            mask = left_active & right_active
            delta = np.abs(left.astype(np.int32) - right.astype(np.int32))[mask]
            result.append(
                {
                    "left_seed": int(left_seed),
                    "right_seed": int(right_seed),
                    "shared_surface_columns": int(delta.size),
                    "differing_surface_fraction": (
                        float(np.count_nonzero(delta) / delta.size)
                        if delta.size
                        else 0.0
                    ),
                    "absolute_height_delta": _numeric_summary(delta),
                }
            )
    return result


def _numeric_summary(values: np.ndarray) -> dict[str, int | float | None]:
    source = np.asarray(values)
    if not source.size:
        return {
            "count": 0,
            "minimum": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "count": int(source.size),
        "minimum": _number(np.min(source)),
        "p05": float(np.percentile(source, 5)),
        "median": float(np.median(source)),
        "mean": float(np.mean(source)),
        "p95": float(np.percentile(source, 95)),
        "maximum": _number(np.max(source)),
    }


def _number(value: Any) -> int | float:
    scalar = np.asarray(value).item()
    return int(scalar) if isinstance(scalar, (int, np.integer)) else float(scalar)


def _asset_key_hex(words: Sequence[int]) -> str:
    return np.asarray(tuple(words), dtype=">u4").tobytes().hex()


def _report_sha256(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("report_semantic_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _measurement_split_contract(
    report: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    override_salt_sha256: str | None,
) -> dict[str, Any]:
    source_contract = _mapping(
        report.get("measurement_split_contract"),
        "source measurement split contract",
    )
    raw_mapping = _mapping(source_contract.get("mapping"), "source seed mapping")
    source_mapping = {
        str(_nonnegative_int(int(seed), "source seed")): _split(value)
        for seed, value in raw_mapping.items()
    }
    seeds = tuple(sorted(int(seed) for seed in source_mapping))
    heldout_count = sum(value == "heldout" for value in source_mapping.values())
    if heldout_count < 1 or len(seeds) - heldout_count < 2:
        raise ValueError("measurement split requires heldout and two calibration seeds")
    source_salt = _sha256(plan.get("split_salt_sha256"), "source split salt")
    if _seed_split(seeds, heldout_count=heldout_count, salt=source_salt) != (
        source_mapping
    ):
        raise ValueError("capture plan salt does not reproduce its seed mapping")
    effective_salt = (
        _sha256(override_salt_sha256, "measurement split override")
        if override_salt_sha256 is not None
        else source_salt
    )
    effective_mapping = _seed_split(
        seeds,
        heldout_count=heldout_count,
        salt=effective_salt,
    )
    return {
        "method": "sha256_seed_rank_v1",
        "source_salt_sha256": source_salt,
        "source_mapping": source_mapping,
        "effective_salt_sha256": effective_salt,
        "effective_mapping": effective_mapping,
        "overrides_source_report": effective_mapping != source_mapping,
        "scope": "seed_identity_shared_across_all_analyzed_structures",
    }


def _seed_split(
    seeds: Sequence[int],
    *,
    heldout_count: int,
    salt: str,
) -> dict[str, str]:
    normalized_salt = _sha256(salt, "seed split salt")
    ranked = sorted(
        (int(seed) for seed in seeds),
        key=lambda seed: (
            hashlib.sha256(
                (
                    "hytalerl_worldgen_v2_seed_split_v1\0"
                    f"{normalized_salt}\0{seed}"
                ).encode()
            ).digest(),
            seed,
        ),
    )
    heldout = set(ranked[:heldout_count])
    return {
        str(seed): "heldout" if seed in heldout else "calibration"
        for seed in seeds
    }


def _split(value: object) -> str:
    if value not in ("calibration", "heldout"):
        raise ValueError("measurement split must be calibration or heldout")
    return str(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in value
    ):
        raise ValueError(f"{label} must be a SHA-256")
    return value.lower()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _string_int_mapping(value: object, label: str) -> dict[str, int]:
    raw = _mapping(value, label)
    result: dict[str, int] = {}
    for key, count in raw.items():
        asset_id = _nonempty_string(key, f"{label} asset ID")
        result[asset_id] = _nonnegative_int(count, f"{label} count")
    return result


def _relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("artifact report path must be relative")
    path = Path(value)
    if any(part == ".." for part in path.parts):
        raise ValueError("artifact report path cannot escape its root")
    return path


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(value, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-report-sha256")
    parser.add_argument("--measurement-split-salt-sha256")
    parser.add_argument("--resolve-block-asset-ids", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = analyze_capture_report(
        args.report,
        output=args.output,
        expected_capture_report_sha256=args.expected_report_sha256,
        measurement_split_salt_sha256=(
            args.measurement_split_salt_sha256
        ),
        resolve_block_asset_ids=args.resolve_block_asset_ids,
    )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "structure_count": len(result["structures"]),
                "artifact_count": sum(
                    int(value["artifact_count"])
                    for value in result["structures"]
                ),
                "report_semantic_sha256": result["report_semantic_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
