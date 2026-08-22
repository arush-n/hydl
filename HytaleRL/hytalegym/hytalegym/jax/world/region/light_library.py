"""Host binding from an exact Region selection to native static light."""

from __future__ import annotations

from pathlib import Path

from hytalegym.jax.world.region.library import (
    RegionLibraryEnvironmentSelection,
)
from hytalegym.jax.world.region.native_light import (
    RegionNativeLightAtlas,
    region_native_light_atlas_from_optional_snapshots,
    region_native_light_atlas_from_snapshots,
)
from hytalegym.worldgen.region import (
    RegionArtifactEntry,
    RegionArtifactLibrary,
    RegionLightCoverageLibrary,
    RegionLightLibrary,
)


def region_native_light_atlas_from_artifact_libraries(
    selection: RegionLibraryEnvironmentSelection,
    *,
    region_manifest_path: str | Path,
    light_manifest_path: str | Path,
) -> RegionNativeLightAtlas:
    """Load light in the exact order of a selected Region working set.

    This host-only constructor verifies the complete Region and light
    manifests before creating device arrays. It must not run inside JIT.
    """

    region_library, entries = _selected_region_entries(
        selection,
        region_manifest_path,
    )
    working = selection.working_set
    capacity = len(entries)
    light_library = RegionLightLibrary.load(
        light_manifest_path,
        region_library,
    )
    snapshots = light_library.load_for_region_entries(entries)
    return region_native_light_atlas_from_snapshots(
        snapshots,
        source_region_semantic_sha256=working.artifact_semantic_sha256,
        world_ids=range(capacity),
    )


def region_native_light_atlas_from_coverage_library(
    selection: RegionLibraryEnvironmentSelection,
    *,
    region_manifest_path: str | Path,
    light_coverage_manifest_path: str | Path,
    expected_light_coverage_semantic_sha256: str,
) -> RegionNativeLightAtlas:
    """Load exact native rows while keeping rejected Regions unavailable."""

    region_library, entries = _selected_region_entries(
        selection,
        region_manifest_path,
    )
    coverage = RegionLightCoverageLibrary.load(
        light_coverage_manifest_path,
        region_library,
    )
    expected_coverage = _sha256(
        expected_light_coverage_semantic_sha256,
        "expected_light_coverage_semantic_sha256",
    )
    if coverage.semantic_sha256 != expected_coverage:
        raise ValueError("Region light coverage semantic SHA-256 changed")
    snapshots = coverage.load_optional_for_region_entries(
        entries,
        region_library=region_library,
    )
    return region_native_light_atlas_from_optional_snapshots(
        snapshots,
        source_region_semantic_sha256=tuple(
            entry.semantic_sha256 for entry in entries
        ),
        core_min_chunk_xz=tuple(entry.core_min_chunk_xz for entry in entries),
        world_ids=range(len(entries)),
    )


def _selected_region_entries(
    selection: RegionLibraryEnvironmentSelection,
    region_manifest_path: str | Path,
) -> tuple[RegionArtifactLibrary, tuple[RegionArtifactEntry, ...]]:
    if not isinstance(selection, RegionLibraryEnvironmentSelection):
        raise TypeError(
            "selection must be a RegionLibraryEnvironmentSelection"
        )
    region_library = RegionArtifactLibrary.load(region_manifest_path)
    working = selection.working_set
    if region_library.semantic_sha256 != working.library_semantic_sha256:
        raise ValueError("Region selection names a different artifact library")
    entries = region_library.select(
        working.split,
        len(working.artifact_semantic_sha256),
        selection_key=working.selection_key,
    )
    if tuple(entry.semantic_sha256 for entry in entries) != (
        working.artifact_semantic_sha256
    ):
        raise ValueError("Region working-set identity differs from its manifest")
    return region_library, entries


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase 64-hex")
    return value


__all__ = [
    "region_native_light_atlas_from_artifact_libraries",
    "region_native_light_atlas_from_coverage_library",
]
