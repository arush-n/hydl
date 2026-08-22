"""Bind exact WorldGen V2 captures to the JAX Region runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache, partial
import hashlib
from importlib import import_module
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from hytalegym.geometry.contract import FLAG_OPAQUE
from hytalegym.jax.combat.arsenal.environment import (
    ArsenalResetBatch,
    geometry_arsenal_world_capabilities,
    geometry_arsenal_world_tokens,
)
from hytalegym.jax.combat.observation.v3.world_tokens import (
    DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
)
from hytalegym.jax.world import WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY
from hytalegym.jax.world.navigation.adapter import (
    native_region_graph_target_navigation_provider,
    surrogate_region_walk_target_navigation_provider,
)
from hytalegym.jax.world.region.geometry import (
    REGION_COMBAT_SWEEP_CELL_CAPACITY,
    region_geometry_from_atlas,
)
from hytalegym.jax.world.region.traversal import (
    REGION_TRAVERSAL_POLICY_STAGING_CAPACITY,
    query_region_traversal_tokens,
)

from arena.publication_worlds import (
    MAX_PUBLICATION_RESIDENT_WORLDS,
    PublishedWorldSpec,
    materialize_publication_worlds,
    publication_info,
    publication_spawns,
)


_ROOT = Path(__file__).resolve().parents[1]
_V2_ROOT = _ROOT / "experimental" / "worldgen-v2"
_CATALOG = _V2_ROOT / "jax_port" / "catalogs" / "worldgen-v2-r1.catalog"
_ARTIFACTS = (
    _V2_ROOT / "evidence" / "region-corpus-0.5.7-0B99A911-five-structures-20-seeds-r1"
)
MAX_WORLD_POOL = MAX_PUBLICATION_RESIDENT_WORLDS


@dataclass(frozen=True, slots=True)
class WorldSpec:
    """One bounded, reproducible WorldGen V2 terrain pool."""

    name: str
    structure: str
    seed: int
    assignment_key: int = 0
    pool_seeds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("world name must be a non-empty label")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("world seed must be an integer")
        if isinstance(self.assignment_key, bool) or not isinstance(
            self.assignment_key, int
        ):
            raise TypeError("world assignment_key must be an integer")
        if not isinstance(self.pool_seeds, tuple) or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in self.pool_seeds
        ):
            raise TypeError("world pool_seeds must be a tuple of integers")
        if len(self.seeds) > MAX_WORLD_POOL:
            raise ValueError(f"world pools are limited to {MAX_WORLD_POOL} seeds")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("world pool seeds must be unique")

    @property
    def seeds(self) -> tuple[int, ...]:
        return (self.seed, *self.pool_seeds)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "generator": "worldgen_v2",
            "structure": self.structure,
            "seed": self.seed,
            "seeds": list(self.seeds),
            "pool_size": len(self.seeds),
            "assignment_key": self.assignment_key,
        }


@dataclass(frozen=True, slots=True)
class WorldBinding:
    providers: Mapping[str, Any]
    identity: Mapping[str, Any]


ArenaWorldSpec = WorldSpec | PublishedWorldSpec


@lru_cache(maxsize=1)
def _bundle():
    # The V2 package uses ``jax_port`` absolute imports internally.  Resolve
    # its owning source root here so Console and local tools do not depend on a
    # caller-specific PYTHONPATH that the WSL training launcher happened to set.
    if str(_V2_ROOT) not in sys.path:
        sys.path.insert(0, str(_V2_ROOT))
    module = import_module("jax_port.bundles.bundle")
    return module.V2JaxBundle.load(_CATALOG, _ARTIFACTS, verify_source=True)


def world_spec_for_split(
    name: str,
    structure: str,
    split: str,
    *,
    count: int = MAX_WORLD_POOL,
    selection_key: int = 0,
) -> WorldSpec:
    """Select a reproducible WorldGen V2 pool from one declared split."""

    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_WORLD_POOL:
        raise ValueError(f"count must be in [1, {MAX_WORLD_POOL}]")
    artifacts = _bundle().select(
        structure,
        split=split,
        count=count,
        selection_key=selection_key,
    )
    seeds = tuple(artifact.seed for artifact in artifacts)
    return WorldSpec(
        name=name,
        structure=structure,
        seed=seeds[0],
        assignment_key=selection_key,
        pool_seeds=seeds[1:],
    )


@lru_cache(maxsize=4)
def _resident_selection(
    structure: str, seeds: tuple[int, ...], assignment_key: int
):
    # Load the bundle first: it puts the V2 source root on sys.path, and it
    # fixes which of the two possible module trees this process is using.
    # The loader must come from that same tree -- importing it as
    # ``experimental.worldgen-v2.jax_port...`` while the bundle came from
    # ``jax_port...`` yields two distinct V2JaxBundle classes, and the
    # isinstance check inside materialize_jax_worlds then rejects a valid
    # bundle with "bundle must be a V2JaxBundle".
    bundle = _bundle()
    loader = import_module("jax_port.bundles.jax_loader")
    return loader.load_selected_jax_worlds(
        bundle,
        structure,
        seeds=seeds,
        environment_count=len(seeds),
        assignment_key=assignment_key,
    )


def _resident_world(spec: WorldSpec):
    """Reuse a physical atlas across cosmetic names and seed ordering."""

    return _resident_selection(
        spec.structure, tuple(sorted(spec.seeds)), spec.assignment_key
    )


def _world_set(spec: WorldSpec, batch: int):
    if batch < 1:
        raise ValueError("world batch must be positive")
    world = _resident_world(spec)
    if batch == len(spec.seeds):
        return world
    indices = jnp.arange(batch, dtype=jnp.int32) % len(spec.seeds)
    return replace(
        world,
        environment_world_id=world.environment_world_id[indices],
        environment_spawn_position=world.environment_spawn_position[indices],
    )


def bind_world(
    spec: ArenaWorldSpec,
    batch: int,
    params: Any,
    config: Any,
) -> WorldBinding:
    """Materialize once on the host, then expose only pure-JAX providers."""

    if isinstance(spec, PublishedWorldSpec):
        return _bind_published_world(spec, batch, params, config)
    if not isinstance(spec, WorldSpec):
        raise TypeError("world spec must be WorldSpec or PublishedWorldSpec")

    worlds = _world_set(spec, batch)
    spawn = worlds.environment_spawn_position.astype(jnp.float32)
    target_spawn = spawn + jnp.asarray(params.target_offset, dtype=jnp.float32)
    providers, navigation_mode, navigation_route_choice = _world_providers(
        worlds,
        params,
        config,
        spawn=spawn,
        target_spawn=target_spawn,
    )

    artifacts = {artifact.seed: artifact for artifact in worlds.artifacts}
    artifact = artifacts[spec.seed]
    pool_artifacts = [
        {
            "seed": seed,
            "split": artifacts[seed].split,
            "region_semantic_sha256": artifacts[seed].region_semantic_sha256,
        }
        for seed in spec.seeds
    ]
    resident_ids = np.asarray(_resident_world(spec).environment_world_id)
    assignment_seed_cycle = [
        worlds.artifacts[int(world_id)].seed for world_id in resident_ids
    ]
    identity = {
        **spec.describe(),
        "split": artifact.split,
        "region_semantic_sha256": artifact.region_semantic_sha256,
        "artifacts": pool_artifacts,
        "pool_semantic_sha256": _pool_digest(pool_artifacts),
        "assignment_seed_cycle": assignment_seed_cycle,
        "bundle_semantic_sha256": worlds.source_bundle_semantic_sha256,
        "target_navigation": navigation_mode,
        "target_navigation_route_choice": navigation_route_choice,
    }
    return WorldBinding(
        providers=MappingProxyType(providers),
        identity=MappingProxyType(identity),
    )


def _world_providers(
    worlds: Any,
    params: Any,
    config: Any,
    *,
    spawn: Any,
    target_spawn: Any,
) -> tuple[dict[str, Any], str, str]:
    """Build the common pure-JAX Region provider set."""

    geometry = region_geometry_from_atlas(
        worlds.physical_atlas,
        worlds.environment_world_id,
        agent_bounds=params.agent_bounds,
        target_bounds=params.target_bounds,
        agent_los_offset=params.agent_eye_offset,
        target_los_offset=params.target_eye_offset,
        max_los_distance=float(params.sensor_range),
        # A 1.805-block NPC plus one 30 Hz fall/knockback step can touch four
        # voxel rows. Three falsely exhausted valid Region motion mid-duel.
        sweep_cell_capacity=REGION_COMBAT_SWEEP_CELL_CAPACITY,
    )
    role_mask = (
        worlds.physical_atlas.cell_flags.astype(jnp.int32) & jnp.int32(FLAG_OPAQUE)
    ) != 0
    capability_provider = partial(
        geometry_arsenal_world_capabilities,
        geometry=geometry,
        config=config,
        role_opaque_mask=role_mask,
    )

    def token_provider(state, combat_params):
        traversal = _agent_traversal_observation(worlds, state)
        return geometry_arsenal_world_tokens(
            state,
            combat_params,
            geometry=geometry,
            role_opaque_mask=role_mask,
            geometry_provenance=WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY,
            traversal=traversal,
            policy_config=DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
        )

    def reset_provider(_keys):
        return ArsenalResetBatch(spawn, target_spawn)

    navigation_provider, navigation_mode, navigation_route_choice = (
        _target_navigation_binding(worlds, geometry)
    )
    return {
        "geometry_provider": geometry,
        "target_navigation_provider": navigation_provider,
        "world_capability_provider": capability_provider,
        "world_token_provider": token_provider,
        "world_geometry_config": DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
        "reset_provider": reset_provider,
    }, navigation_mode, navigation_route_choice


def _agent_traversal_observation(worlds: Any, state: Any) -> Any:
    if worlds.traversal_atlas is None:
        return None
    return query_region_traversal_tokens(
        worlds.traversal_atlas,
        worlds.environment_world_id,
        state.combat.position[:, :1, :],
        token_capacity=REGION_TRAVERSAL_POLICY_STAGING_CAPACITY,
        max_distance=DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG.maximum_distance,
    )


def _target_navigation_binding(
    worlds: Any,
    geometry: Any,
) -> tuple[Any, str, str]:
    traversal = worlds.traversal_atlas
    if traversal is None:
        return (
            surrogate_region_walk_target_navigation_provider,
            "surrogate_region_walk_over_exact_target_geometry",
            "surrogate_greedy_not_native_certified",
        )
    if not _native_graph_profiles_match(
        traversal,
        worlds.environment_world_id,
        geometry.target_bounds,
    ):
        return (
            surrogate_region_walk_target_navigation_provider,
            "surrogate_region_walk_native_graph_profile_mismatch",
            "surrogate_greedy_not_native_certified",
        )
    return (
        partial(
            native_region_graph_target_navigation_provider,
            traversal=traversal,
            environment_world_id=worlds.environment_world_id,
        ),
        "native_graph_guided_exact_walk_substeps",
        "bounded_local_greedy_not_native_AStar",
    )


def _native_graph_profiles_match(
    traversal: Any,
    environment_world_id: Any,
    target_bounds: Any,
) -> bool:
    graph_mask = np.asarray(traversal.graph_mask, dtype=np.bool_)
    graph_world_id = np.asarray(traversal.world_id, dtype=np.int32)
    graph_bounds = np.asarray(traversal.actor_bounds, dtype=np.float32)
    world_ids = np.asarray(environment_world_id, dtype=np.int32)
    bounds = np.asarray(target_bounds, dtype=np.float32)
    if bounds.shape == (1, 6):
        bounds = np.broadcast_to(bounds, (len(world_ids), 6))
    if bounds.shape != (len(world_ids), 6):
        return False
    for world_id, expected in zip(world_ids, bounds, strict=True):
        selected = graph_mask & (graph_world_id == world_id)
        if not np.any(selected) or not np.allclose(
            graph_bounds[selected],
            expected,
            rtol=0.0,
            atol=1.0e-6,
        ):
            return False
    return True


def _bind_published_world(
    spec: PublishedWorldSpec,
    batch: int,
    params: Any,
    config: Any,
) -> WorldBinding:
    """Bind one verified publication window without loading the full corpus."""

    resident = materialize_publication_worlds(spec, batch)
    worlds = resident.batch.worlds
    spawn, target_spawn, agent_team, target_team = publication_spawns(
        spec,
        resident,
        params.target_offset,
    )
    providers, navigation_mode, navigation_route_choice = _world_providers(
        worlds,
        params,
        config,
        spawn=spawn,
        target_spawn=target_spawn,
    )

    pool_artifacts = [
        {
            "seed": artifact.seed,
            "split": artifact.split,
            "region_semantic_sha256": artifact.region_semantic_sha256,
        }
        for artifact in worlds.artifacts
    ]
    info = publication_info(spec)
    publication_seeds = info["seeds"]
    identity = {
        **info,
        # ``seeds`` means resident worlds in the original WorldSpec identity;
        # preserve it and expose the complete publication separately.
        "publication_seeds": publication_seeds,
        "seeds": list(resident.window_seeds),
        "pool_size": len(resident.window_seeds),
        "resident_slots": resident.resident_capacity,
        "structure": resident.manifest["world_structure_asset_id"],
        "seed": resident.window_seeds[0],
        "split": worlds.artifacts[0].split,
        "region_semantic_sha256": worlds.artifacts[0].region_semantic_sha256,
        "artifacts": pool_artifacts,
        "pool_semantic_sha256": _pool_digest(pool_artifacts),
        "assignment_seed_cycle": list(resident.resident_seed_cycle),
        "bundle_semantic_sha256": resident.bundle_semantic_sha256,
        "traversal_pack_semantic_sha256": resident.traversal_semantic_sha256,
        "structure_pack_semantic_sha256": resident.structure_semantic_sha256,
        "agent_traversal_tokens": "native_exact_graph",
        "agent_team_id": agent_team,
        "target_team_id": target_team,
        "target_navigation": navigation_mode,
        "target_navigation_route_choice": navigation_route_choice,
    }
    return WorldBinding(
        providers=MappingProxyType(providers),
        identity=MappingProxyType(identity),
    )


def _pool_digest(artifacts: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\0".join(
            f"{row['seed']}:{row['region_semantic_sha256']}" for row in artifacts
        ).encode()
    ).hexdigest()


__all__ = [
    "ArenaWorldSpec",
    "MAX_WORLD_POOL",
    "PublishedWorldSpec",
    "WorldBinding",
    "WorldSpec",
    "bind_world",
    "world_spec_for_split",
]
