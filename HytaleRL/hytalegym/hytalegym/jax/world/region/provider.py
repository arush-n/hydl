"""Host-side exact Region provider discovery with fail-closed fallback."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import operator
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from hytalegym.jax.world.region.library import (
    REGION_ARTIFACT_USAGE_HELDOUT_EVALUATION,
    REGION_ARTIFACT_USAGE_TRAINING,
    RegionLibraryEnvironmentSelection,
    region_environments_from_artifact_manifest,
)
from hytalegym.jax.world.region.types import RegionAtlas, RegionTraversalAtlas
from hytalegym.worldgen.region import (
    REGION_ARTIFACT_LIBRARY_SCHEMA,
    REGION_TRAVERSAL_LIBRARY_SCHEMA,
)

REGION_PROVIDER_SELECTION_SCHEMA = "hytalerl_region_provider_selection_v1"
REGION_PROVIDER_SELECTION_VERSION = 1
REGION_PROVIDER_ARTIFACT_ROOT_ENV = "HYTALERL_WORLD_ARTIFACT_ROOT"
REGION_PROVIDER_NATIVE = "native_region_artifact"
REGION_PROVIDER_SURROGATE = "surrogate_fail_closed"

_USAGES = frozenset(
    {
        REGION_ARTIFACT_USAGE_TRAINING,
        REGION_ARTIFACT_USAGE_HELDOUT_EVALUATION,
    }
)


@dataclass(frozen=True)
class RegionProviderEnvironmentSelection:
    """Exact native selection or an explicit unavailable-native fallback."""

    provider: str
    source_library_semantic_sha256: str
    provider_available: jax.Array
    environment_world_id: jax.Array
    reason: str
    native_region: RegionLibraryEnvironmentSelection | None

    @property
    def atlas(self) -> RegionAtlas | None:
        if self.native_region is None:
            return None
        return self.native_region.atlas

    @property
    def traversal_atlas(self) -> RegionTraversalAtlas | None:
        if self.native_region is None:
            return None
        return self.native_region.traversal_atlas


def region_or_surrogate_environments(
    source_library_semantic_sha256: str,
    *,
    usage: str,
    working_set_capacity: int,
    environment_count: int,
    selection_key: int,
    assignment_key: int,
    artifact_root: str | Path | None = None,
    require_traversal: bool = True,
    cell_palette_capacity: int | None = None,
    shape_palette_capacity: int | None = None,
    traversal_node_capacity: int | None = None,
    traversal_column_node_capacity: int | None = None,
) -> RegionProviderEnvironmentSelection:
    """Resolve an exact native library without per-caller manifest wiring.

    Discovery is host-only. No configured root, no exact library, or no
    required traversal sidecar returns the explicit surrogate control with all
    native availability false. Invalid or ambiguous configured artifacts raise.
    """

    source = _sha256(
        source_library_semantic_sha256,
        "source_library_semantic_sha256",
    )
    if usage not in _USAGES:
        raise ValueError("usage must be training or heldout_evaluation")
    working_capacity = _positive_int(
        working_set_capacity,
        "working_set_capacity",
    )
    environments = _positive_int(environment_count, "environment_count")
    selection = _nonnegative_int(selection_key, "selection_key")
    assignment = _nonnegative_int(assignment_key, "assignment_key")
    if not isinstance(require_traversal, bool):
        raise TypeError("require_traversal must be bool")
    for value, label in (
        (cell_palette_capacity, "cell_palette_capacity"),
        (shape_palette_capacity, "shape_palette_capacity"),
        (traversal_node_capacity, "traversal_node_capacity"),
        (
            traversal_column_node_capacity,
            "traversal_column_node_capacity",
        ),
    ):
        if value is not None:
            _positive_int(value, label)

    root = _configured_root(artifact_root)
    if root is None:
        return _fallback(source, environments, "artifact_root_not_configured")
    region_manifest = _unique_manifest(
        root,
        filename="manifest.json",
        schema=REGION_ARTIFACT_LIBRARY_SCHEMA,
        semantic_field="library_semantic_sha256",
        semantic_sha256=source,
        label="Region library",
    )
    if region_manifest is None:
        return _fallback(source, environments, "source_library_not_available")

    traversal_manifest = _unique_manifest(
        root,
        filename="traversal-manifest.json",
        schema=REGION_TRAVERSAL_LIBRARY_SCHEMA,
        semantic_field="source_region_library_semantic_sha256",
        semantic_sha256=source,
        label="Region traversal library",
    )
    if require_traversal and traversal_manifest is None:
        return _fallback(source, environments, "traversal_library_not_available")

    traversal_semantic = None
    if traversal_manifest is not None:
        traversal_semantic = _sha256(
            _load_json(traversal_manifest).get("library_semantic_sha256"),
            "traversal library semantic SHA-256",
        )
    native = region_environments_from_artifact_manifest(
        region_manifest,
        usage=usage,
        expected_library_semantic_sha256=source,
        working_set_capacity=working_capacity,
        environment_count=environments,
        selection_key=selection,
        assignment_key=assignment,
        cell_palette_capacity=cell_palette_capacity,
        shape_palette_capacity=shape_palette_capacity,
        traversal_manifest_path=traversal_manifest,
        expected_traversal_library_semantic_sha256=traversal_semantic,
        traversal_node_capacity=traversal_node_capacity,
        traversal_column_node_capacity=traversal_column_node_capacity,
    )
    return RegionProviderEnvironmentSelection(
        provider=REGION_PROVIDER_NATIVE,
        source_library_semantic_sha256=source,
        provider_available=jnp.ones((environments,), dtype=jnp.bool_),
        environment_world_id=native.environment_world_id,
        reason="exact_source_semantic_match",
        native_region=native,
    )


def region_provider_selection_contract() -> dict[str, object]:
    """Return the checkpoint-pinnable provider-discovery boundary."""

    return {
        "schema": REGION_PROVIDER_SELECTION_SCHEMA,
        "version": REGION_PROVIDER_SELECTION_VERSION,
        "source_identity": "exact_region_library_semantic_sha256",
        "artifact_root": {
            "explicit_parameter": True,
            "environment_fallback": REGION_PROVIDER_ARTIFACT_ROOT_ENV,
            "repository_layout_assumption": None,
            "excluded_historical_directory": "world-evidence-history",
        },
        "native_requirements": [
            "one_exact_region_library",
            "one_exact_source_bound_traversal_library_when_required",
            "all_existing_library_and_graph_integrity_gates",
        ],
        "surrogate_fail_closed_reasons": [
            "artifact_root_not_configured",
            "source_library_not_available",
            "traversal_library_not_available",
        ],
        "configured_invalid_or_ambiguous": "raise_before_publication",
        "native_availability": "per_environment_true_only_after_full_load",
        "fallback_availability": "per_environment_false_with_world_id_minus_one",
        "runtime": "host_discovery_then_fixed_shape_jax_arrays",
        "seed_regeneration": "never",
    }


def region_provider_selection_contract_sha256() -> str:
    payload = json.dumps(
        region_provider_selection_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _fallback(
    source: str,
    environment_count: int,
    reason: str,
) -> RegionProviderEnvironmentSelection:
    return RegionProviderEnvironmentSelection(
        provider=REGION_PROVIDER_SURROGATE,
        source_library_semantic_sha256=source,
        provider_available=jnp.zeros((environment_count,), dtype=jnp.bool_),
        environment_world_id=jnp.full(
            (environment_count,),
            -1,
            dtype=jnp.int32,
        ),
        reason=reason,
        native_region=None,
    )


def _configured_root(value: str | Path | None) -> Path | None:
    configured: str | Path | None = value
    if configured is None:
        configured = os.environ.get(REGION_PROVIDER_ARTIFACT_ROOT_ENV)
    if configured is None:
        return None
    if isinstance(configured, str) and not configured.strip():
        raise ValueError("artifact root must not be empty")
    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"artifact root is not a directory: {root}")
    return root


def _unique_manifest(
    root: Path,
    *,
    filename: str,
    schema: str,
    semantic_field: str,
    semantic_sha256: str,
    label: str,
) -> Path | None:
    matches = []
    for path in sorted(root.rglob(filename)):
        if "world-evidence-history" in path.relative_to(root).parts:
            continue
        value = _load_json(path)
        if value.get("schema") != schema:
            continue
        semantic = _sha256(value.get(semantic_field), semantic_field)
        if semantic == semantic_sha256:
            matches.append(path.resolve())
    if len(matches) > 1:
        rendered = ", ".join(str(path) for path in matches)
        raise ValueError(f"{label} source identity is ambiguous: {rendered}")
    return matches[0] if matches else None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot inspect provider manifest: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"provider manifest must be an object: {path}")
    return value


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
    "REGION_PROVIDER_ARTIFACT_ROOT_ENV",
    "REGION_PROVIDER_NATIVE",
    "REGION_PROVIDER_SELECTION_SCHEMA",
    "REGION_PROVIDER_SELECTION_VERSION",
    "REGION_PROVIDER_SURROGATE",
    "RegionProviderEnvironmentSelection",
    "region_or_surrogate_environments",
    "region_provider_selection_contract",
    "region_provider_selection_contract_sha256",
]
