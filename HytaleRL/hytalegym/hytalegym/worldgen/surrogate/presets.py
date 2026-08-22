"""Versioned primary-zone inputs for non-authoritative surrogate terrain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hytalegym.worldgen.surrogate.assets import WorldStructureProfile
from hytalegym.worldgen.surrogate.contract import TERRAIN_PARAMETER_COUNT

PRIMARY_ZONE_PRESET_VERSION = 1


@dataclass(frozen=True)
class PrimaryZoneSurrogatePreset:
    """Explicit geometry and material choices; never native density output."""

    profile_id: str
    profile_sha256: str
    biome_ids: tuple[str, ...]
    parameter_rows: tuple[tuple[float, ...], ...]
    surface_materials: tuple[str, ...]
    subsurface_materials: tuple[str, ...]

    @property
    def biome_parameters(self) -> dict[str, np.ndarray]:
        return {
            biome_id: np.asarray(row, dtype=np.float32)
            for biome_id, row in zip(
                self.biome_ids,
                self.parameter_rows,
                strict=True,
            )
        }

    @property
    def surface_asset_ids(self) -> dict[str, str]:
        return dict(zip(self.biome_ids, self.surface_materials, strict=True))

    @property
    def subsurface_asset_ids(self) -> dict[str, str]:
        return dict(zip(self.biome_ids, self.subsurface_materials, strict=True))


def primary_zone_v1_preset(
    profile: WorldStructureProfile | str,
) -> PrimaryZoneSurrogatePreset:
    """Return one hash-checked primary-zone preset or reject the profile."""

    if isinstance(profile, str):
        preset = _PRESETS_BY_ID.get(profile)
        if preset is None:
            raise ValueError(f"unsupported primary-zone profile {profile!r}")
        return preset
    if not isinstance(profile, WorldStructureProfile):
        raise TypeError("profile must be a WorldStructureProfile or profile ID")
    preset = _PRESETS_BY_SHA256.get(profile.provenance.content_sha256)
    if preset is None:
        raise ValueError("world profile has no primary-zone v1 surrogate preset")
    biome_ids = tuple(biome.biome for biome in profile.biomes)
    if biome_ids != preset.biome_ids:
        raise ValueError("world profile biomes differ from surrogate preset")
    return preset


def _preset(
    profile_id: str,
    profile_sha256: str,
    rows: tuple[tuple[str, tuple[float, ...], str, str], ...],
) -> PrimaryZoneSurrogatePreset:
    parameters = tuple(_parameter_row(values) for _, values, _, _ in rows)
    return PrimaryZoneSurrogatePreset(
        profile_id=profile_id,
        profile_sha256=profile_sha256,
        biome_ids=tuple(biome_id for biome_id, _, _, _ in rows),
        parameter_rows=parameters,
        surface_materials=tuple(surface for _, _, surface, _ in rows),
        subsurface_materials=tuple(subsurface for _, _, _, subsurface in rows),
    )


def _parameter_row(values: tuple[float, ...]) -> tuple[float, ...]:
    if len(values) != 11:
        raise ValueError("primary-zone active parameter rows require 11 values")
    return tuple(float(value) for value in values) + (0.0,) * (
        TERRAIN_PARAMETER_COUNT - len(values)
    )


_PRESETS = (
    _preset(
        "Zone1_Plains1",
        "9d314de90286e622b1950130515a52d75fcd8aa69cc6c33984daf2dfebbaa5e3",
        (
            (
                "Plains1_Oak",
                (100, 12, 6, 2, 3, 42, 8, 0.22, 72, 7, 0.42),
                "Soil_Grass",
                "Rock_Marble",
            ),
            (
                "Plains1_Gorges",
                (108, 30, 13, 4, 16, 48, 12, 0.05, 76, 9, 0.30),
                "Soil_Grass",
                "Rock_Stone",
            ),
            (
                "Plains1_Deeproot",
                (96, 13, 8, 3, 5, 34, 16, -0.05, 68, 18, 0.08),
                "Soil_Grass_Deep",
                "Rock_Stone",
            ),
            (
                "Plains1_River",
                (94, 3, 2, 1, 0, 0, 0, 1, 0, 0, 1),
                "Soil_Mud",
                "Rock_Marble",
            ),
            (
                "Plains1_Shore",
                (98, 4, 2, 1, 0, 0, 0, 1, 0, 0, 1),
                "Soil_Sand",
                "Rock_Stone",
            ),
            ("Oceans", (84, 6, 3, 1, 0, 0, 0, 1, 0, 0, 1), "Rock_Stone", "Rock_Stone"),
        ),
    ),
    _preset(
        "Zone2_Desert1",
        "c9969a818bdd512ca502a1bbca50464ae1ac7c72ee7625cbc6f41581377f8359",
        (
            (
                "Desert1_Rocky",
                (110, 24, 11, 3, 12, 44, 10, 0.18, 72, 8, 0.48),
                "Soil_Gravel_Sand_White",
                "Rock_Sandstone",
            ),
            (
                "Desert1_Stacks",
                (114, 20, 9, 3, 24, 46, 9, 0.25, 74, 7, 0.52),
                "Soil_Clay_Yellow",
                "Rock_Sandstone",
            ),
            (
                "Desert1_River",
                (98, 4, 2, 1, 0, 0, 0, 1, 0, 0, 1),
                "Soil_Clay_Orange",
                "Rock_Sandstone",
            ),
            (
                "Desert1_Shore",
                (99, 5, 2, 1, 0, 0, 0, 1, 0, 0, 1),
                "Soil_Sand_White",
                "Rock_Sandstone",
            ),
            ("Oceans", (84, 6, 3, 1, 0, 0, 0, 1, 0, 0, 1), "Rock_Stone", "Rock_Stone"),
        ),
    ),
    _preset(
        "Zone3_Taiga1",
        "65a50a4643e8779eb8617eebc10d2096c066281fa8da39e4db1a5f1d640e921e",
        (
            (
                "Taiga1_Mountains",
                (112, 34, 13, 4, 18, 42, 11, 0.15, 74, 8, 0.48),
                "Soil_Snow",
                "Rock_Quartzite",
            ),
            (
                "Taiga1_Redwood",
                (108, 14, 7, 2, 4, 40, 9, 0.25, 70, 7, 0.55),
                "Soil_Needles",
                "Rock_Quartzite",
            ),
            (
                "Taiga1_River",
                (98, 4, 2, 1, 0, 0, 0, 1, 0, 0, 1),
                "Soil_Gravel_Sand",
                "Rock_Quartzite",
            ),
            (
                "Taiga1_Shore",
                (100, 5, 3, 1, 0, 0, 0, 1, 0, 0, 1),
                "Soil_Gravel_Sand",
                "Rock_Quartzite",
            ),
            ("Oceans", (84, 6, 3, 1, 0, 0, 0, 1, 0, 0, 1), "Rock_Stone", "Rock_Stone"),
        ),
    ),
    _preset(
        "Zone4_Volcanic1",
        "f1b6ecab3b7df1b1a9b86f57ac6157348f916dc939023fc1566e0599191131ab",
        (
            (
                "Volcanic1_Caldera",
                (122, 20, 12, 4, 24, 48, 14, 0.08, 82, 9, 0.38),
                "Soil_Ash",
                "Rock_Volcanic",
            ),
            (
                "Volcanic1_Jungle",
                (118, 16, 8, 3, 7, 42, 12, 0.14, 76, 8, 0.44),
                "Soil_Grass",
                "Rock_Stone",
            ),
            (
                "Volcanic1_River",
                (101, 6, 3, 2, 2, 40, 8, 0.28, 0, 0, 1),
                "Soil_Sand_Ashen",
                "Rock_Volcanic",
            ),
            (
                "Volcanic1_Shore",
                (100, 7, 4, 2, 5, 0, 0, 1, 0, 0, 1),
                "Soil_Sand_Ashen",
                "Rock_Slate",
            ),
            ("Oceans", (84, 6, 3, 1, 0, 0, 0, 1, 0, 0, 1), "Rock_Stone", "Rock_Stone"),
        ),
    ),
)
_PRESETS_BY_ID = {preset.profile_id: preset for preset in _PRESETS}
_PRESETS_BY_SHA256 = {preset.profile_sha256: preset for preset in _PRESETS}


__all__ = [
    "PRIMARY_ZONE_PRESET_VERSION",
    "PrimaryZoneSurrogatePreset",
    "primary_zone_v1_preset",
]
