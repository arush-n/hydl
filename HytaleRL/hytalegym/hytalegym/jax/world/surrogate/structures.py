"""Host planning adapter from JAX terrain columns to prefab instances."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.surrogate.terrain import (
    TerrainGeneratorConfig,
    sample_surrogate_terrain_columns,
)
from hytalegym.worldgen.surrogate import (
    GeneratedStructurePlan,
    StructurePlacementConfig,
    SurrogateBiomePrefabLayer,
    SurrogateStructureCatalog,
    SurrogateWorldCapacity,
    plan_biome_prefab_tile,
    plan_structure_tile,
)


def plan_structure_tile_from_terrain(
    catalog: SurrogateStructureCatalog,
    seed_words: tuple[int, int],
    *,
    tile_min_xz: tuple[int, int],
    terrain_config: TerrainGeneratorConfig,
    placement_config: StructurePlacementConfig | None = None,
    capacity: SurrogateWorldCapacity | None = None,
) -> GeneratedStructurePlan:
    """Plan an offline tile using the same globally keyed terrain kernel."""

    if not isinstance(terrain_config, TerrainGeneratorConfig):
        raise TypeError("terrain_config must be a TerrainGeneratorConfig")
    sample = _terrain_column_sampler(seed_words, terrain_config)
    return plan_structure_tile(
        catalog,
        seed_words,
        tile_min_xz=tile_min_xz,
        column_sampler=sample,
        config=placement_config,
        capacity=capacity,
    )


def plan_biome_prefab_tile_from_terrain(
    layers: tuple[SurrogateBiomePrefabLayer, ...],
    seed_words: tuple[int, int],
    *,
    tile_min_xz: tuple[int, int],
    terrain_config: TerrainGeneratorConfig,
    capacity: SurrogateWorldCapacity | None = None,
) -> GeneratedStructurePlan:
    """Plan source-bound biome prefab layers from generated columns."""

    if not isinstance(terrain_config, TerrainGeneratorConfig):
        raise TypeError("terrain_config must be a TerrainGeneratorConfig")
    return plan_biome_prefab_tile(
        layers,
        seed_words,
        tile_min_xz=tile_min_xz,
        column_sampler=_terrain_column_sampler(seed_words, terrain_config),
        capacity=capacity,
    )


def _terrain_column_sampler(
    seed_words: tuple[int, int],
    terrain_config: TerrainGeneratorConfig,
):
    def sample(
        positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if positions.shape == (0, 2):
            return (
                np.zeros(0, dtype=np.bool_),
                np.zeros(0, dtype=np.int16),
                np.zeros(0, dtype=np.uint8),
            )
        result = sample_surrogate_terrain_columns(
            jnp.asarray([seed_words], dtype=jnp.uint32),
            jnp.asarray(positions[None, ...], dtype=jnp.int32),
            terrain_config,
        )
        return tuple(
            np.asarray(jax.device_get(value[0]))
            for value in (
                result.valid,
                result.terrain_height,
                result.biome_code,
            )
        )

    return sample


__all__ = [
    "plan_biome_prefab_tile_from_terrain",
    "plan_structure_tile_from_terrain",
]
