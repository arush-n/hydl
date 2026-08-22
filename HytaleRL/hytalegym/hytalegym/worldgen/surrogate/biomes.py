"""Provenance-locked biome inputs for explicitly surrogate terrain."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from hytalegym.worldgen.surrogate.assets import (
    BIOME_PREFIX,
    DEFAULT_GENERATOR_NODE_CAPACITY,
    BiomeGeneratorAsset,
    HytaleAssetArchive,
    HytaleAssetError,
    WorldStructureProfile,
)
from hytalegym.worldgen.surrogate.presets import primary_zone_v1_preset


@dataclass(frozen=True)
class LocalBiomeCatalog:
    profile: WorldStructureProfile
    biomes: tuple[BiomeGeneratorAsset, ...]


@dataclass(frozen=True)
class SurrogateBiomeMaterialBinding:
    biome_id: str
    surface_asset_id: str
    subsurface_asset_id: str
    source_fluid_materials: tuple[str, ...]
    biome_sha256: str


@dataclass(frozen=True)
class SurrogateBiomeMaterialPlan:
    """Explicit material choices; native density/material ASTs are not executed."""

    bindings: tuple[SurrogateBiomeMaterialBinding, ...]
    fluid_mode: str

    @property
    def surface_asset_ids(self) -> dict[str, str]:
        return {binding.biome_id: binding.surface_asset_id for binding in self.bindings}

    @property
    def subsurface_asset_ids(self) -> dict[str, str]:
        return {
            binding.biome_id: binding.subsurface_asset_id for binding in self.bindings
        }


def load_local_biome_catalog(
    archive: HytaleAssetArchive,
    profile: WorldStructureProfile,
    *,
    node_capacity: int = DEFAULT_GENERATOR_NODE_CAPACITY,
) -> LocalBiomeCatalog:
    """Resolve every profile biome to exactly one local generator asset."""

    if not isinstance(archive, HytaleAssetArchive):
        raise TypeError("archive must be a HytaleAssetArchive")
    if not isinstance(profile, WorldStructureProfile):
        raise TypeError("profile must be a WorldStructureProfile")
    assets: list[BiomeGeneratorAsset] = []
    for biome in profile.biomes:
        if any(character in biome.biome for character in "/\\"):
            raise HytaleAssetError(f"unsafe biome ID {biome.biome!r}")
        matches = archive.entry_paths(
            prefix=BIOME_PREFIX,
            suffix=f"/{biome.biome}.json",
        )
        if len(matches) != 1:
            raise HytaleAssetError(
                f"biome {biome.biome!r} resolved to {len(matches)} entries"
            )
        asset = archive.load_biome_generator(
            matches[0],
            node_capacity=node_capacity,
        )
        if asset.biome_id != biome.biome:
            raise HytaleAssetError("resolved biome asset ID does not match profile")
        assets.append(asset)
    return LocalBiomeCatalog(profile=profile, biomes=tuple(assets))


def bind_surrogate_biome_materials(
    catalog: LocalBiomeCatalog,
    *,
    surface_asset_ids: Mapping[str, str],
    subsurface_asset_ids: Mapping[str, str],
    fluid_mode: str,
) -> SurrogateBiomeMaterialPlan:
    """Validate explicit surrogate choices against exact local material leaves."""

    if not isinstance(catalog, LocalBiomeCatalog):
        raise TypeError("catalog must be a LocalBiomeCatalog")
    if fluid_mode != "disabled":
        raise ValueError("only explicit fluid_mode='disabled' is supported")
    expected = {asset.biome_id for asset in catalog.biomes}
    if set(surface_asset_ids) != expected:
        raise ValueError("surface material bindings must match catalog biomes")
    if set(subsurface_asset_ids) != expected:
        raise ValueError("subsurface material bindings must match catalog biomes")

    bindings: list[SurrogateBiomeMaterialBinding] = []
    for asset in catalog.biomes:
        if asset.terrain_type != "DAOTerrain":
            raise HytaleAssetError(
                f"unsupported biome terrain type {asset.terrain_type!r}"
            )
        if asset.unsupported_top_level_fields:
            raise HytaleAssetError(
                f"biome {asset.biome_id!r} has unsupported fields: "
                + ", ".join(asset.unsupported_top_level_fields)
            )
        surface = _selected_material(
            surface_asset_ids[asset.biome_id],
            asset,
            "surface",
        )
        subsurface = _selected_material(
            subsurface_asset_ids[asset.biome_id],
            asset,
            "subsurface",
        )
        bindings.append(
            SurrogateBiomeMaterialBinding(
                biome_id=asset.biome_id,
                surface_asset_id=surface,
                subsurface_asset_id=subsurface,
                source_fluid_materials=asset.fluid_materials,
                biome_sha256=asset.provenance.content_sha256,
            )
        )
    return SurrogateBiomeMaterialPlan(
        bindings=tuple(bindings),
        fluid_mode=fluid_mode,
    )


def bind_primary_zone_v1_materials(
    catalog: LocalBiomeCatalog,
) -> SurrogateBiomeMaterialPlan:
    """Bind versioned surrogate choices after exact asset-membership checks."""

    if not isinstance(catalog, LocalBiomeCatalog):
        raise TypeError("catalog must be a LocalBiomeCatalog")
    preset = primary_zone_v1_preset(catalog.profile)
    return bind_surrogate_biome_materials(
        catalog,
        surface_asset_ids=preset.surface_asset_ids,
        subsurface_asset_ids=preset.subsurface_asset_ids,
        fluid_mode="disabled",
    )


def _selected_material(
    value: str,
    asset: BiomeGeneratorAsset,
    label: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} material ID must be a non-empty string")
    if value == "Empty":
        raise HytaleAssetError(f"{label} material cannot be Empty")
    if value not in asset.solid_materials:
        raise HytaleAssetError(
            f"{label} material {value!r} is absent from biome {asset.biome_id!r}"
        )
    return value


__all__ = [
    "LocalBiomeCatalog",
    "SurrogateBiomeMaterialBinding",
    "SurrogateBiomeMaterialPlan",
    "bind_primary_zone_v1_materials",
    "bind_surrogate_biome_materials",
    "load_local_biome_catalog",
]
