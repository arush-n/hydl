"""Bounded Arena windows over immutable WorldGen V2 publications.

A publication may expose hundreds or thousands of exact seed captures. Loading
all of them into one JAX atlas would turn corpus breadth into a memory leak, so
Arena selects one deterministic resident window at a time. The complete
publication remains the identity and allowlist; only the selected window's
physical, block, fluid, traversal, structure, and recipe arrays become
resident.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import import_module
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import jax.numpy as jnp


MAX_PUBLICATION_RESIDENT_WORLDS = 4
PUBLICATION_RESIDENT_CACHE_SIZE = 2


@dataclass(frozen=True, slots=True)
class PublishedWorldSpec:
    """One deterministic resident window from an exact V2 publication.

    ``window_index`` pages through the publication's sorted seed allowlist.
    ``resident_capacity`` controls device memory and is independent of both the
    publication size and the Arena batch size. The final, short page is padded
    with inactive atlas slots so its leading Region shape remains stable.
    """

    name: str
    publication: Path | str
    resident_capacity: int = MAX_PUBLICATION_RESIDENT_WORLDS
    window_index: int = 0
    assignment_key: int | None = None
    agent_team_id: str | None = None
    target_team_id: str | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("world name must be a non-empty label")
        path = Path(self.publication).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"WorldGen V2 publication is unavailable: {path}")
        object.__setattr__(self, "publication", path)
        _positive_int(self.resident_capacity, "resident_capacity")
        if self.resident_capacity > MAX_PUBLICATION_RESIDENT_WORLDS:
            raise ValueError(
                "publication resident windows are limited to "
                f"{MAX_PUBLICATION_RESIDENT_WORLDS} worlds"
            )
        _nonnegative_int(self.window_index, "window_index")
        if self.assignment_key is not None:
            _nonnegative_int(self.assignment_key, "assignment_key")
        for value, label in (
            (self.agent_team_id, "agent_team_id"),
            (self.target_team_id, "target_team_id"),
        ):
            if value is not None and (not value or value != value.strip()):
                raise ValueError(f"{label} must be a non-empty label when set")
        if (
            self.agent_team_id is not None
            and self.agent_team_id == self.target_team_id
        ):
            raise ValueError("agent_team_id and target_team_id must differ")

    def with_window(self, window_index: int) -> "PublishedWorldSpec":
        """Return the same immutable publication at another resident page."""

        return replace(self, window_index=window_index)

    def describe(self) -> dict[str, Any]:
        """Return validated publication and selected-window metadata."""

        return publication_info(self)


@dataclass(frozen=True, slots=True)
class PublicationResidentBatch:
    """Materialized resident arrays plus publication provenance."""

    batch: Any
    manifest: Mapping[str, Any]
    all_seed_count: int
    window_count: int
    window_index: int
    window_seeds: tuple[int, ...]
    resident_seed_cycle: tuple[int, ...]
    resident_capacity: int
    assignment_key: int
    bundle_semantic_sha256: str
    traversal_semantic_sha256: str
    structure_semantic_sha256: str


@dataclass(frozen=True, slots=True)
class _PublicationSource:
    path: Path
    manifest: Mapping[str, Any]
    bundle: Any
    recipe: Any
    traversal: Any
    structures: Any
    artifacts: tuple[Any, ...]


def publication_info(spec: PublishedWorldSpec) -> dict[str, Any]:
    """Describe a publication without materializing Region cell arrays."""

    if not isinstance(spec, PublishedWorldSpec):
        raise TypeError("spec must be a PublishedWorldSpec")
    source = _source_for(spec.publication)
    selected, windows = _window(
        source.artifacts,
        capacity=spec.resident_capacity,
        index=spec.window_index,
    )
    assignment = _assignment_key(spec, source.manifest)
    return {
        "name": spec.name,
        "generator": "worldgen_v2_publication",
        "environment_id": source.manifest["environment_id"],
        "recipe_id": source.recipe.recipe_id,
        "world_structure_asset_id": source.manifest[
            "world_structure_asset_id"
        ],
        "publication_semantic_sha256": source.manifest[
            "publication_semantic_sha256"
        ],
        "recipe_semantic_sha256": source.recipe.semantic_sha256,
        "bundle_semantic_sha256": source.bundle.semantic_sha256,
        "seed_count": len(source.artifacts),
        "seeds": [artifact.seed for artifact in source.artifacts],
        "resident_capacity": spec.resident_capacity,
        "window_count": windows,
        "window_index": spec.window_index,
        "resident_seeds": [artifact.seed for artifact in selected],
        "assignment_key": assignment,
        "team_ids": [team.team_id for team in source.recipe.teams],
        "agent_team_id": spec.agent_team_id,
        "target_team_id": spec.target_team_id,
        "required_capabilities": sorted(source.recipe.required_capabilities),
        "objective_kind": source.recipe.objective.kind,
    }


def materialize_publication_worlds(
    spec: PublishedWorldSpec,
    batch: int,
) -> PublicationResidentBatch:
    """Materialize one bounded resident page and cycle it across ``batch``."""

    if not isinstance(spec, PublishedWorldSpec):
        raise TypeError("spec must be a PublishedWorldSpec")
    environments = _positive_int(batch, "world batch")
    path = Path(spec.publication)
    stat = path.stat()
    resident = _resident_window(
        str(path),
        stat.st_mtime_ns,
        stat.st_size,
        spec.resident_capacity,
        spec.window_index,
        spec.assignment_key,
    )
    if environments == spec.resident_capacity:
        return resident

    indices = jnp.arange(environments, dtype=jnp.int32) % spec.resident_capacity
    worlds = replace(
        resident.batch.worlds,
        environment_world_id=resident.batch.worlds.environment_world_id[indices],
        environment_spawn_position=(
            resident.batch.worlds.environment_spawn_position[indices]
        ),
    )
    runtime = resident.batch.runtime._replace(
        **{
            field: getattr(resident.batch.runtime, field)[indices]
            for field in resident.batch.runtime._fields
        }
    )
    return replace(
        resident,
        batch=replace(resident.batch, worlds=worlds, runtime=runtime),
    )


def publication_spawns(
    spec: PublishedWorldSpec,
    resident: PublicationResidentBatch,
    target_offset: Any,
) -> tuple[Any, Any, str, str | None]:
    """Resolve Arena's two actor spawns from the publication team contract."""

    team_ids = resident.batch.team_ids
    if not team_ids:
        raise ValueError("publication recipe declares no teams")
    agent_index = _team_index(team_ids, spec.agent_team_id, fallback=0)
    target_index = None
    if spec.target_team_id is not None:
        target_index = _team_index(team_ids, spec.target_team_id, fallback=None)
    elif len(team_ids) > 1:
        target_index = 1 if agent_index != 1 else 0
    if target_index == agent_index:
        raise ValueError("publication agent and target resolve to the same team")

    agent = resident.batch.runtime.team_spawn_position[:, agent_index]
    target = (
        agent + jnp.asarray(target_offset, dtype=jnp.float32)
        if target_index is None
        else resident.batch.runtime.team_spawn_position[:, target_index]
    )
    return (
        agent.astype(jnp.float32),
        target.astype(jnp.float32),
        team_ids[agent_index],
        None if target_index is None else team_ids[target_index],
    )


def publication_cache_info() -> dict[str, Any]:
    """Expose bounded host/device cache state for diagnostics and tests."""

    source = _load_source.cache_info()
    resident = _resident_window.cache_info()
    return {
        "source": source._asdict(),
        "resident": resident._asdict(),
        "resident_limit": PUBLICATION_RESIDENT_CACHE_SIZE,
        "resident_world_limit": MAX_PUBLICATION_RESIDENT_WORLDS,
    }


def clear_publication_cache() -> None:
    """Drop publication objects and resident JAX arrays held by this process."""

    _resident_window.cache_clear()
    _load_source.cache_clear()


def _source_for(path: Path | str) -> _PublicationSource:
    source = Path(path).resolve()
    stat = source.stat()
    return _load_source(str(source), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=16)
def _load_source(
    path_text: str,
    publication_mtime_ns: int,
    publication_size: int,
) -> _PublicationSource:
    del publication_mtime_ns, publication_size
    path = Path(path_text)
    publication = import_module(
        "experimental.worldgen-v2.custom_environments.publication"
    )
    bundle_module = import_module("experimental.worldgen-v2.jax_port.bundles.bundle")
    recipe_module = import_module(
        "experimental.worldgen-v2.jax_port.worlds.environment_recipe"
    )
    traversal_module = import_module(
        "experimental.worldgen-v2.jax_port.bundles.traversal_pack"
    )
    structure_module = import_module(
        "experimental.worldgen-v2.jax_port.structures.structure_pack"
    )

    manifest = publication.load_custom_environment_publication(path)
    root = path.parent
    capture_root = _contained_directory(root, manifest["native_capture"]["root"])
    bundle_path = _reference_path(root, manifest["jax_bundle"])
    bundle = bundle_module.V2JaxBundle.load(
        bundle_path,
        capture_root,
        verify_source=True,
        verify_artifacts=False,
    )
    if bundle.semantic_sha256 != manifest["jax_bundle"]["semantic_sha256"]:
        raise ValueError("publication JAX bundle semantic changed")

    seeds = tuple(int(seed) for seed in manifest["seeds"])
    artifacts = bundle.select(
        manifest["world_structure_asset_id"],
        seeds=seeds,
    )
    recipe = recipe_module.V2EnvironmentRecipe.load(
        _reference_path(root, manifest["environment_recipe"]),
        bundle,
    )
    if recipe.semantic_sha256 != manifest["environment_recipe"][
        "semantic_sha256"
    ]:
        raise ValueError("publication environment recipe semantic changed")
    if recipe.artifacts != artifacts:
        raise ValueError("publication recipe selects different exact worlds")

    traversal = traversal_module.V2TraversalPack.load(
        _reference_path(root, manifest["traversal"]["pack"]),
        bundle,
        verify_graphs=False,
    )
    if traversal.semantic_sha256 != manifest["traversal"]["pack"][
        "semantic_sha256"
    ] or not traversal.covers(artifacts):
        raise ValueError("publication traversal pack identity or coverage changed")

    structures = structure_module.V2StructurePack.load(
        _reference_path(root, manifest["structures"]["pack"]),
        bundle,
        verify_snapshots=False,
    )
    if structures.semantic_sha256 != manifest["structures"]["pack"][
        "semantic_sha256"
    ] or not structures.covers(artifacts):
        raise ValueError("publication structure pack identity or coverage changed")
    if structures.registry.semantic_sha256 != manifest["structures"][
        "registry_semantic_sha256"
    ]:
        raise ValueError("publication structure registry semantic changed")

    materialization = manifest["materialization"]
    if int(materialization["exact_world_count"]) != len(artifacts):
        raise ValueError("publication exact-world census changed")
    return _PublicationSource(
        path=path,
        manifest=manifest,
        bundle=bundle,
        recipe=recipe,
        traversal=traversal,
        structures=structures,
        artifacts=artifacts,
    )


@lru_cache(maxsize=PUBLICATION_RESIDENT_CACHE_SIZE)
def _resident_window(
    path_text: str,
    publication_mtime_ns: int,
    publication_size: int,
    resident_capacity: int,
    window_index: int,
    assignment_key: int | None,
) -> PublicationResidentBatch:
    source = _load_source(
        path_text,
        publication_mtime_ns,
        publication_size,
    )
    selected, windows = _window(
        source.artifacts,
        capacity=resident_capacity,
        index=window_index,
    )
    assignment = (
        int(source.manifest["materialization"]["assignment_key"])
        if assignment_key is None
        else assignment_key
    )
    recipe = replace(source.recipe, artifacts=selected)
    recipe_module = import_module(
        "experimental.worldgen-v2.jax_port.worlds.environment_recipe"
    )
    materialization = source.manifest["materialization"]
    traversal_capacity = _second_dimension(
        materialization.get("traversal_node_shape")
    )
    structure_capacity = _second_dimension(
        materialization.get("structure_instance_shape")
    )
    batch = recipe_module.materialize_environment_recipe(
        recipe,
        source.bundle,
        # Always expose a full page. A final short page cycles its real worlds
        # while inactive Region slots keep the atlas leading shape unchanged.
        environment_count=resident_capacity,
        assignment_key=assignment,
        region_capacity=resident_capacity,
        traversal_pack=source.traversal,
        traversal_node_capacity=traversal_capacity,
        structure_pack=source.structures,
        structure_instance_capacity=structure_capacity,
    )
    resident_seed_cycle = tuple(
        selected[int(world_id)].seed
        for world_id in batch.worlds.environment_world_id.tolist()
    )
    return PublicationResidentBatch(
        batch=batch,
        manifest=source.manifest,
        all_seed_count=len(source.artifacts),
        window_count=windows,
        window_index=window_index,
        window_seeds=tuple(artifact.seed for artifact in selected),
        resident_seed_cycle=resident_seed_cycle,
        resident_capacity=resident_capacity,
        assignment_key=assignment,
        bundle_semantic_sha256=source.bundle.semantic_sha256,
        traversal_semantic_sha256=source.traversal.semantic_sha256,
        structure_semantic_sha256=source.structures.semantic_sha256,
    )


def _window(
    artifacts: tuple[Any, ...],
    *,
    capacity: int,
    index: int,
) -> tuple[tuple[Any, ...], int]:
    if not artifacts:
        raise ValueError("publication has no exact worlds")
    resident = _positive_int(capacity, "resident_capacity")
    page = _nonnegative_int(index, "window_index")
    windows = math.ceil(len(artifacts) / resident)
    if page >= windows:
        raise ValueError(
            f"window_index {page} is outside publication window count {windows}"
        )
    start = page * resident
    return artifacts[start : start + resident], windows


def _assignment_key(
    spec: PublishedWorldSpec,
    manifest: Mapping[str, Any],
) -> int:
    return (
        int(manifest["materialization"]["assignment_key"])
        if spec.assignment_key is None
        else spec.assignment_key
    )


def _team_index(
    team_ids: tuple[str, ...],
    requested: str | None,
    *,
    fallback: int | None,
) -> int:
    if requested is None:
        if fallback is None:
            raise ValueError("a publication target team is required")
        return fallback
    try:
        return team_ids.index(requested)
    except ValueError as error:
        raise ValueError(
            f"publication recipe has no team {requested!r}; have {list(team_ids)}"
        ) from error


def _reference_path(root: Path, reference: Mapping[str, Any]) -> Path:
    return _contained_path(root, reference["path"], directory=None)


def _contained_directory(root: Path, relative: Any) -> Path:
    return _contained_path(root, relative, directory=True)


def _contained_path(
    root: Path,
    relative: Any,
    *,
    directory: bool | None,
) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("publication path must be a non-empty string")
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or relative != posix.as_posix():
        raise ValueError("publication path must be normalized and relative")
    candidate = root.joinpath(*posix.parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("publication path escapes its root") from error
    if directory is True and not candidate.is_dir():
        raise FileNotFoundError(f"publication directory is unavailable: {relative}")
    if directory is None and not candidate.is_file():
        raise FileNotFoundError(f"publication file is unavailable: {relative}")
    return candidate


def _second_dimension(value: Any) -> int | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    return _positive_int(value[1], "publication fixed-shape capacity")


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


__all__ = [
    "MAX_PUBLICATION_RESIDENT_WORLDS",
    "PUBLICATION_RESIDENT_CACHE_SIZE",
    "PublicationResidentBatch",
    "PublishedWorldSpec",
    "clear_publication_cache",
    "materialize_publication_worlds",
    "publication_cache_info",
    "publication_info",
    "publication_spawns",
]
