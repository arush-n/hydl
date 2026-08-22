"""Fixed-working-set publication from an exact host Region library."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import operator
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.region.action_runtime import (
    RegionActionRuntimeState,
    mutable_block_state_from_region_snapshots,
)
from hytalegym.jax.world.region.atlas import region_atlas_from_snapshots
from hytalegym.jax.world.region.block_semantics import (
    region_block_semantic_atlas_from_snapshots,
    region_block_semantic_atlas_without_evidence,
)
from hytalegym.jax.world.region.mutable_geometry import (
    bind_region_mutable_blocks,
)
from hytalegym.jax.world.region.traversal import (
    region_traversal_atlas_from_graphs,
)
from hytalegym.jax.world.region.types import (
    RegionGeometryState,
    RegionAtlas,
    RegionTraversalAtlas,
)
from hytalegym.worldgen.region import (
    RegionArtifactLibrary,
    RegionBlockSemanticLibrary,
    RegionTraversalGraphLibrary,
)

REGION_ARTIFACT_USAGE_TRAINING = "training"
REGION_ARTIFACT_USAGE_HELDOUT_EVALUATION = "heldout_evaluation"
REGION_ARTIFACT_ASSIGNMENT_METHOD = "sha256_balanced_cycle_v1"
_USAGE_SPLIT = {
    REGION_ARTIFACT_USAGE_TRAINING: "train",
    REGION_ARTIFACT_USAGE_HELDOUT_EVALUATION: "heldout",
}


@dataclass(frozen=True)
class RegionLibraryAtlasSelection:
    """Audit metadata paired with one shape-stable device working set."""

    atlas: RegionAtlas
    library_semantic_sha256: str
    split: str
    selection_key: int
    artifact_semantic_sha256: tuple[str, ...]
    artifact_seed: tuple[int, ...]
    traversal_atlas: RegionTraversalAtlas | None = None
    # How the working set was chosen, not merely what it resolved to. The
    # runtime re-derives the same set later to bind sidecars and mutable
    # physics, and a set composed by seed cannot be re-derived from
    # `selection_key`. ``None`` means the key ranking selected it.
    selection_seeds: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RegionLibraryEnvironmentSelection:
    """One split-pinned working set and automatic environment assignments."""

    working_set: RegionLibraryAtlasSelection
    usage: str
    assignment_key: int
    environment_world_id: jax.Array

    @property
    def atlas(self) -> RegionAtlas:
        return self.working_set.atlas

    @property
    def traversal_atlas(self) -> RegionTraversalAtlas | None:
        return self.working_set.traversal_atlas


def region_action_runtime_from_artifact_libraries(
    selection: RegionLibraryEnvironmentSelection,
    geometry: RegionGeometryState,
    *,
    region_manifest_path: str | Path,
    block_semantic_manifest_path: str | Path | None,
    mutation_capacity: int,
) -> RegionActionRuntimeState:
    """Bind sidecars and mutable physics before compiled reset/rollout.

    This host-only constructor performs manifest I/O and identity checks.
    Close over its returned PyTree; never invoke it from a JIT trace.
    """

    if not isinstance(selection, RegionLibraryEnvironmentSelection):
        raise TypeError(
            "selection must be a RegionLibraryEnvironmentSelection"
        )
    if not isinstance(geometry, RegionGeometryState):
        raise TypeError("geometry must be a RegionGeometryState")
    if not np.array_equal(
        np.asarray(jax.device_get(geometry.environment_world_id)),
        np.asarray(jax.device_get(selection.environment_world_id)),
    ):
        raise ValueError("geometry environment assignment differs from selection")

    library = RegionArtifactLibrary.load(region_manifest_path)
    working = selection.working_set
    if library.semantic_sha256 != working.library_semantic_sha256:
        raise ValueError("Region selection names a different artifact library")
    capacity = len(working.artifact_semantic_sha256)
    seeds = working.selection_seeds
    # Re-derive the *same* working set the atlas was published from. A set
    # composed by seed is not reachable from `selection_key`, which can only
    # express "the N that hash first", so re-deriving by key here would bind
    # sidecars and mutable physics to worlds the atlas does not hold -- and it
    # would do it silently, because every shape still matches.
    if seeds is None:
        entries = library.select(
            working.split,
            capacity,
            selection_key=working.selection_key,
        )
    else:
        entries = library.select_seeds(working.split, seeds)
    if tuple(entry.semantic_sha256 for entry in entries) != (
        working.artifact_semantic_sha256
    ):
        raise ValueError("Region working-set identity differs from its manifest")
    if seeds is None:
        snapshots = library.load_selection(
            working.split,
            capacity,
            selection_key=working.selection_key,
        )
    else:
        snapshots = library.load_seed_selection(working.split, seeds)
    if block_semantic_manifest_path is None:
        # A library with terrain and traversal but no block-semantic sidecars.
        # The runtime is still built -- the mutable physics half is what the
        # rollout needs -- but every semantic mask is false, so the block heads
        # close instead of reading identities that were never captured.
        semantic_atlas = region_block_semantic_atlas_without_evidence(
            geometry.atlas,
        )
    else:
        semantic_library = RegionBlockSemanticLibrary.load(
            block_semantic_manifest_path,
            library,
        )
        semantic_snapshots = semantic_library.load_for_region_entries(entries)
        semantic_atlas = region_block_semantic_atlas_from_snapshots(
            semantic_snapshots,
            snapshots,
            geometry.atlas,
            world_ids=range(capacity),
        )
    mutable = mutable_block_state_from_region_snapshots(
        snapshots,
        geometry,
        world_ids=range(capacity),
        mutation_capacity=mutation_capacity,
    )
    return RegionActionRuntimeState(
        bind_region_mutable_blocks(geometry, mutable),
        semantic_atlas,
    )


def region_atlas_from_artifact_library(
    library: RegionArtifactLibrary,
    *,
    split: str,
    working_set_capacity: int,
    selection_key: int,
    selection_seeds: Sequence[int] | None = None,
    cell_palette_capacity: int | None = None,
    shape_palette_capacity: int | None = None,
    traversal_library: RegionTraversalGraphLibrary | None = None,
    traversal_node_capacity: int | None = None,
    traversal_column_node_capacity: int | None = None,
) -> RegionLibraryAtlasSelection:
    """Publish exactly ``working_set_capacity`` worlds without seed generation.

    ``selection_seeds`` names the working set outright and takes precedence over
    ``selection_key``. It exists because the key is a digest *rank* over the whole
    split, so it can only express "the N that hash first" -- a composed set such
    as "half flat, half varied" is not reachable through it at any key. When the
    seeds are given, ``selection_key`` is still recorded on the result so the
    artifact keeps saying which key it would otherwise have used.
    """

    if not isinstance(library, RegionArtifactLibrary):
        raise TypeError("library must be a RegionArtifactLibrary")
    requested: tuple[int, ...] | None = None
    if selection_seeds is None:
        entries = library.select(
            split,
            working_set_capacity,
            selection_key=selection_key,
        )
        snapshots = library.load_selection(
            split,
            working_set_capacity,
            selection_key=selection_key,
        )
    else:
        requested = tuple(int(seed) for seed in selection_seeds)
        # A shorter list would leave the trailing atlas slots holding whatever
        # padding the capacity implies, and `environment_world_id` would still
        # route environments to them.
        if len(requested) != working_set_capacity:
            raise ValueError(
                "explicit Region selection must match working_set_capacity: "
                f"{len(requested)} seeds against capacity {working_set_capacity}"
            )
        entries = library.select_seeds(split, requested)
        snapshots = library.load_seed_selection(split, requested)
    atlas = region_atlas_from_snapshots(
        snapshots,
        world_ids=range(len(entries)),
        region_capacity=working_set_capacity,
        cell_palette_capacity=cell_palette_capacity,
        shape_palette_capacity=shape_palette_capacity,
        validate_overlaps=False,
    )
    traversal_atlas = None
    if traversal_library is not None:
        if not isinstance(
            traversal_library,
            RegionTraversalGraphLibrary,
        ):
            raise TypeError(
                "traversal_library must be a RegionTraversalGraphLibrary"
            )
        if (
            traversal_library.manifest[
                "source_region_library_semantic_sha256"
            ]
            != library.semantic_sha256
        ):
            raise ValueError("terrain and traversal libraries differ")
        graphs = traversal_library.load_for_region_entries(entries)
        traversal_atlas = region_traversal_atlas_from_graphs(
            atlas,
            graphs,
            node_capacity=traversal_node_capacity,
            column_node_capacity=traversal_column_node_capacity,
        )
    return RegionLibraryAtlasSelection(
        atlas=atlas,
        library_semantic_sha256=library.semantic_sha256,
        split=split,
        selection_key=selection_key,
        artifact_semantic_sha256=tuple(
            entry.semantic_sha256 for entry in entries
        ),
        artifact_seed=tuple(entry.seed for entry in entries),
        traversal_atlas=traversal_atlas,
        selection_seeds=requested,
    )


def region_environments_from_artifact_manifest(
    manifest_path: str | Path,
    *,
    usage: str,
    expected_library_semantic_sha256: str,
    working_set_capacity: int,
    environment_count: int,
    selection_key: int,
    assignment_key: int,
    selection_seeds: Sequence[int] | None = None,
    cell_palette_capacity: int | None = None,
    shape_palette_capacity: int | None = None,
    traversal_manifest_path: str | Path | None = None,
    expected_traversal_library_semantic_sha256: str | None = None,
    traversal_node_capacity: int | None = None,
    traversal_column_node_capacity: int | None = None,
) -> RegionLibraryEnvironmentSelection:
    """Load one pinned split and assign environments without naming artifacts."""

    try:
        split = _USAGE_SPLIT[usage]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "usage must be training or heldout_evaluation"
        ) from error
    expected = _sha256(
        expected_library_semantic_sha256,
        "expected_library_semantic_sha256",
    )
    environments = _positive_int(environment_count, "environment_count")
    assignment = _nonnegative_int(assignment_key, "assignment_key")
    library = RegionArtifactLibrary.load(manifest_path)
    if library.semantic_sha256.lower() != expected:
        raise ValueError("Region artifact library semantic SHA-256 changed")
    if (traversal_manifest_path is None) != (
        expected_traversal_library_semantic_sha256 is None
    ):
        raise ValueError(
            "traversal manifest and expected semantic hash are required together"
        )
    traversal_library = None
    if traversal_manifest_path is not None:
        traversal_library = RegionTraversalGraphLibrary.load(
            traversal_manifest_path,
            library,
        )
        expected_traversal = _sha256(
            expected_traversal_library_semantic_sha256,
            "expected traversal library semantic SHA-256",
        )
        if traversal_library.semantic_sha256 != expected_traversal:
            raise ValueError("Region traversal library semantic SHA-256 changed")
    working_set = region_atlas_from_artifact_library(
        library,
        split=split,
        working_set_capacity=working_set_capacity,
        selection_key=selection_key,
        selection_seeds=selection_seeds,
        cell_palette_capacity=cell_palette_capacity,
        shape_palette_capacity=shape_palette_capacity,
        traversal_library=traversal_library,
        traversal_node_capacity=traversal_node_capacity,
        traversal_column_node_capacity=traversal_column_node_capacity,
    )
    ranked_world_ids = sorted(
        range(len(working_set.artifact_semantic_sha256)),
        key=lambda world_id: (
            hashlib.sha256(
                (
                    f"{REGION_ARTIFACT_ASSIGNMENT_METHOD}\0{assignment}\0"
                    f"{working_set.artifact_semantic_sha256[world_id]}"
                ).encode()
            ).digest(),
            working_set.artifact_semantic_sha256[world_id],
        ),
    )
    environment_world_id = jnp.asarray(
        [
            ranked_world_ids[index % len(ranked_world_ids)]
            for index in range(environments)
        ],
        dtype=jnp.int32,
    )
    return RegionLibraryEnvironmentSelection(
        working_set=working_set,
        usage=usage,
        assignment_key=assignment,
        environment_world_id=environment_world_id,
    )


def _positive_int(value: object, label: str) -> int:
    result = _exact_int(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    result = _exact_int(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _exact_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{label} must be a SHA-256")
    return value.lower()


__all__ = [
    "REGION_ARTIFACT_ASSIGNMENT_METHOD",
    "REGION_ARTIFACT_USAGE_HELDOUT_EVALUATION",
    "REGION_ARTIFACT_USAGE_TRAINING",
    "RegionLibraryAtlasSelection",
    "RegionLibraryEnvironmentSelection",
    "region_action_runtime_from_artifact_libraries",
    "region_atlas_from_artifact_library",
    "region_environments_from_artifact_manifest",
]
