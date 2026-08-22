"""Deterministic JAX terrain for non-authoritative surrogate training worlds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.worldgen.region import (
    CAPTURE_BLOCKS_PER_AXIS,
    CHUNK_SIZE,
    MIN_Y,
    WORLD_HEIGHT,
)
from hytalegym.worldgen.surrogate import (
    BIOME_CAPACITY,
    CAVE_SPAN_CAPACITY,
    TERRAIN_PARAMETER_COUNT,
    WorldStructureProfile,
)
from hytalegym.worldgen.surrogate.biomes import (
    LocalBiomeCatalog,
    SurrogateBiomeMaterialPlan,
    bind_primary_zone_v1_materials,
)
from hytalegym.worldgen.surrogate.presets import (
    PRIMARY_ZONE_PRESET_VERSION,
    primary_zone_v1_preset,
)

DEFAULT_BIOME_CODE = 255
SUPPORTED_TERRAIN_PARAMETERS = 11
SURROGATE_TERRAIN_SOURCE_SCHEMA = "hytalerl_surrogate_terrain_source_v1"
PROFILE_ONLY_TERRAIN_SOURCE = "profile_only_uncalibrated"
PRIMARY_ZONE_V1_TERRAIN_SOURCE = "primary_zone_v1_local_assets"

HEIGHT_PARAMETER_COUNT = 5
_CAVE_PARAMETERS_PER_SPAN = 3
_CAVE_PARAMETER_START = 5

_MACRO_SCALE = 256
_MESO_SCALE = 80
_DETAIL_SCALE = 24
_RIDGE_SCALE = 48
_BIOME_LARGE_SCALE = 512
_BIOME_SMALL_SCALE = 160
_CAVE_MASK_SCALE = (72, 44)
_CAVE_FLOOR_SCALE = (120, 88)
_CAVE_HEIGHT_SCALE = (48, 36)
_INT32_MAX_TILE_ORIGIN = np.iinfo(np.int32).max - (CAPTURE_BLOCKS_PER_AXIS - 1)
_AXIS = jnp.arange(CAPTURE_BLOCKS_PER_AXIS, dtype=jnp.int32)


class TerrainGeneratorConfig(NamedTuple):
    parameters: jax.Array
    biome_parameters: jax.Array
    biome_minimum: jax.Array
    biome_maximum: jax.Array
    biome_mask: jax.Array
    surface_cell_code: jax.Array
    subsurface_cell_code: jax.Array
    default_surface_cell_code: jax.Array
    default_subsurface_cell_code: jax.Array


@dataclass(frozen=True)
class SurrogateTerrainBiomeSource:
    """Exact local biome facts referenced by one surrogate terrain config."""

    biome_id: str
    entry_path: str
    content_sha256: str
    prop_graph_sha256: tuple[str, ...]
    surface_asset_id: str
    subsurface_asset_id: str
    omitted_fluid_asset_ids: tuple[str, ...]


@dataclass(frozen=True)
class SurrogateTerrainSource:
    """Versioned provenance boundary; prop execution is never implied."""

    schema: str
    mode: str
    profile_entry_path: str
    profile_sha256: str
    preset_id: str | None
    preset_version: int
    native_calibrated: bool
    fluid_mode: str
    prop_execution_supported: bool
    biomes: tuple[SurrogateTerrainBiomeSource, ...]


@dataclass(frozen=True)
class TerrainGeneratorSpec:
    """Host provenance paired with the numeric JAX generator config."""

    config: TerrainGeneratorConfig
    biome_names: tuple[str, ...]
    profile_sha256: str
    source: SurrogateTerrainSource


class SurrogateTerrainTile(NamedTuple):
    """Complete fixed-shape terrain columns for one or more halo tiles."""

    valid: jax.Array
    seed_words: jax.Array
    tile_min_xz: jax.Array
    parameters: jax.Array
    biome_parameters: jax.Array
    column_known: jax.Array
    terrain_height: jax.Array
    surface_cell_code: jax.Array
    subsurface_cell_code: jax.Array
    biome_code: jax.Array
    cave_span_mask: jax.Array
    cave_min_y: jax.Array
    cave_max_y: jax.Array


class SurrogateTerrainColumnSample(NamedTuple):
    """Globally keyed fixed-query columns used by offline structure planning."""

    valid: jax.Array
    terrain_height: jax.Array
    surface_cell_code: jax.Array
    subsurface_cell_code: jax.Array
    biome_code: jax.Array


class _TerrainFields(NamedTuple):
    valid: jax.Array
    height: jax.Array
    surface: jax.Array
    subsurface: jax.Array
    biome_code: jax.Array
    cave_mask: jax.Array
    cave_min_y: jax.Array
    cave_max_y: jax.Array


def default_terrain_parameters() -> np.ndarray:
    """Return the v2 uncalibrated fallback terrain parameter vector."""

    result = np.zeros(TERRAIN_PARAMETER_COUNT, dtype=np.float32)
    result[:HEIGHT_PARAMETER_COUNT] = [100.0, 18.0, 7.0, 2.0, 4.0]
    return result


def plains_v1_biome_parameters() -> dict[str, np.ndarray]:
    """Return explicit local-scale-informed Plains surrogate presets."""

    return primary_zone_v1_biome_parameters("Zone1_Plains1")


def primary_zone_v1_biome_parameters(
    profile: WorldStructureProfile | str,
) -> dict[str, np.ndarray]:
    """Return local-scale-informed geometry for one exact primary profile."""

    return primary_zone_v1_preset(profile).biome_parameters


def primary_zone_v1_terrain_generator_spec(
    catalog: LocalBiomeCatalog,
    material_plan: SurrogateBiomeMaterialPlan,
    *,
    surface_cell_codes: Mapping[str, int],
    subsurface_cell_codes: Mapping[str, int],
    default_surface_cell_code: int,
    default_subsurface_cell_code: int,
    parameters: Sequence[float] | None = None,
) -> TerrainGeneratorSpec:
    """Build a provenance-complete primary-zone v1 surrogate terrain spec."""

    if not isinstance(catalog, LocalBiomeCatalog):
        raise TypeError("catalog must be a LocalBiomeCatalog")
    if not isinstance(material_plan, SurrogateBiomeMaterialPlan):
        raise TypeError("material_plan must be a SurrogateBiomeMaterialPlan")
    expected_materials = bind_primary_zone_v1_materials(catalog)
    if material_plan != expected_materials:
        raise ValueError("material plan differs from primary-zone v1 bindings")
    preset = primary_zone_v1_preset(catalog.profile)
    spec = terrain_generator_spec_from_profile(
        catalog.profile,
        surface_cell_codes=surface_cell_codes,
        subsurface_cell_codes=subsurface_cell_codes,
        default_surface_cell_code=default_surface_cell_code,
        default_subsurface_cell_code=default_subsurface_cell_code,
        parameters=parameters,
        biome_parameters=preset.biome_parameters,
    )
    sources = tuple(
        SurrogateTerrainBiomeSource(
            biome_id=biome.biome_id,
            entry_path=biome.provenance.entry_path,
            content_sha256=biome.provenance.content_sha256,
            prop_graph_sha256=tuple(
                prop.graph_sha256 for prop in biome.prop_assignments
            ),
            surface_asset_id=binding.surface_asset_id,
            subsurface_asset_id=binding.subsurface_asset_id,
            omitted_fluid_asset_ids=binding.source_fluid_materials,
        )
        for biome, binding in zip(
            catalog.biomes,
            material_plan.bindings,
            strict=True,
        )
    )
    return replace(
        spec,
        source=SurrogateTerrainSource(
            schema=SURROGATE_TERRAIN_SOURCE_SCHEMA,
            mode=PRIMARY_ZONE_V1_TERRAIN_SOURCE,
            profile_entry_path=catalog.profile.provenance.entry_path,
            profile_sha256=catalog.profile.provenance.content_sha256,
            preset_id="primary_zone_v1",
            preset_version=PRIMARY_ZONE_PRESET_VERSION,
            native_calibrated=False,
            fluid_mode=material_plan.fluid_mode,
            prop_execution_supported=False,
            biomes=sources,
        ),
    )


def terrain_generator_spec_from_profile(
    profile: WorldStructureProfile,
    *,
    surface_cell_codes: Mapping[str, int],
    subsurface_cell_codes: Mapping[str, int],
    default_surface_cell_code: int,
    default_subsurface_cell_code: int,
    parameters: Sequence[float] | None = None,
    biome_parameters: Mapping[str, Sequence[float]] | None = None,
) -> TerrainGeneratorSpec:
    """Build a padded numeric config from one validated local asset profile."""

    if not isinstance(profile, WorldStructureProfile):
        raise TypeError("profile must be a WorldStructureProfile")
    if profile.profile_type != "NoiseRange":
        raise ValueError(
            f"unsupported world structure profile type {profile.profile_type!r}"
        )
    if profile.unsupported_fields:
        raise ValueError(
            "world structure profile contains unsupported fields: "
            + ", ".join(profile.unsupported_fields)
        )
    if len(profile.biomes) > BIOME_CAPACITY:
        raise ValueError("world profile exceeds biome capacity")
    if not profile.biomes:
        raise ValueError("world profile contains no biome ranges")
    vector = (
        default_terrain_parameters()
        if parameters is None
        else np.asarray(parameters, dtype=np.float32)
    )
    if vector.shape != (TERRAIN_PARAMETER_COUNT,):
        raise ValueError(
            f"terrain parameters must have shape ({TERRAIN_PARAMETER_COUNT},)"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("terrain parameters must be finite")
    _validate_parameter_vector(vector, "terrain parameters")
    supplied_biomes = {} if biome_parameters is None else dict(biome_parameters)
    expected_biomes = {biome.biome for biome in profile.biomes}
    if supplied_biomes and set(supplied_biomes) != expected_biomes:
        raise ValueError("biome parameter bindings must match profile biomes")

    minimum = np.zeros(BIOME_CAPACITY, dtype=np.float32)
    maximum = np.zeros(BIOME_CAPACITY, dtype=np.float32)
    mask = np.zeros(BIOME_CAPACITY, dtype=np.bool_)
    surface = np.zeros(BIOME_CAPACITY, dtype=np.uint16)
    subsurface = np.zeros(BIOME_CAPACITY, dtype=np.uint16)
    geometry = np.zeros(
        (BIOME_CAPACITY, TERRAIN_PARAMETER_COUNT),
        dtype=np.float32,
    )
    names: list[str] = []
    for index, biome in enumerate(profile.biomes):
        names.append(biome.biome)
        minimum[index] = biome.minimum
        maximum[index] = biome.maximum
        mask[index] = True
        surface[index] = _cell_code(
            surface_cell_codes,
            biome.biome,
            "surface",
        )
        subsurface[index] = _cell_code(
            subsurface_cell_codes,
            biome.biome,
            "subsurface",
        )
        biome_vector = (
            vector
            if not supplied_biomes
            else np.asarray(
                supplied_biomes[biome.biome],
                dtype=np.float32,
            )
        )
        if biome_vector.shape != (TERRAIN_PARAMETER_COUNT,):
            raise ValueError(
                f"biome parameters for {biome.biome!r} must have shape "
                f"({TERRAIN_PARAMETER_COUNT},)"
            )
        _validate_parameter_vector(
            biome_vector,
            f"biome parameters for {biome.biome!r}",
        )
        geometry[index] = biome_vector

    config = TerrainGeneratorConfig(
        parameters=jnp.asarray(vector),
        biome_parameters=jnp.asarray(geometry),
        biome_minimum=jnp.asarray(minimum),
        biome_maximum=jnp.asarray(maximum),
        biome_mask=jnp.asarray(mask),
        surface_cell_code=jnp.asarray(surface),
        subsurface_cell_code=jnp.asarray(subsurface),
        default_surface_cell_code=jnp.asarray(
            _uint16(
                default_surface_cell_code,
                "default surface cell code",
            ),
            dtype=jnp.uint16,
        ),
        default_subsurface_cell_code=jnp.asarray(
            _uint16(
                default_subsurface_cell_code,
                "default subsurface cell code",
            ),
            dtype=jnp.uint16,
        ),
    )
    return TerrainGeneratorSpec(
        config=config,
        biome_names=tuple(names),
        profile_sha256=profile.provenance.content_sha256,
        source=SurrogateTerrainSource(
            schema=SURROGATE_TERRAIN_SOURCE_SCHEMA,
            mode=PROFILE_ONLY_TERRAIN_SOURCE,
            profile_entry_path=profile.provenance.entry_path,
            profile_sha256=profile.provenance.content_sha256,
            preset_id=None,
            preset_version=0,
            native_calibrated=False,
            fluid_mode="disabled",
            prop_execution_supported=False,
            biomes=(),
        ),
    )


def generate_surrogate_terrain(
    seed_words: jax.Array,
    tile_min_xz: jax.Array,
    config: TerrainGeneratorConfig,
) -> SurrogateTerrainTile:
    """Generate globally keyed terrain; incomplete/invalid tiles fail closed."""

    seeds = jnp.asarray(seed_words, dtype=jnp.uint32)
    origins = jnp.asarray(tile_min_xz, dtype=jnp.int32)
    if seeds.ndim != 2 or seeds.shape[1] != 2:
        raise ValueError("seed_words must have shape [batch, 2]")
    if origins.shape != seeds.shape:
        raise ValueError("tile_min_xz must have shape [batch, 2]")
    _validate_config_shapes(config)

    batch = seeds.shape[0]
    origin_valid = jnp.all(
        (origins <= jnp.int32(_INT32_MAX_TILE_ORIGIN))
        & (jnp.mod(origins, jnp.int32(CHUNK_SIZE)) == 0),
        axis=1,
    )
    safe_origins = jnp.where(origin_valid[:, None], origins, 0)
    x = jnp.broadcast_to(
        safe_origins[:, 0, None, None] + _AXIS[None, :, None],
        (batch, CAPTURE_BLOCKS_PER_AXIS, CAPTURE_BLOCKS_PER_AXIS),
    )
    z = jnp.broadcast_to(
        safe_origins[:, 1, None, None] + _AXIS[None, None, :],
        x.shape,
    )

    fields = _terrain_fields(seeds, x, z, config)
    valid = origin_valid & jnp.all(fields.valid, axis=(1, 2))
    known = jnp.broadcast_to(
        valid[:, None, None],
        fields.height.shape,
    )
    return SurrogateTerrainTile(
        valid=valid,
        seed_words=seeds,
        tile_min_xz=origins,
        parameters=jnp.broadcast_to(
            config.parameters,
            (batch, TERRAIN_PARAMETER_COUNT),
        ),
        biome_parameters=jnp.broadcast_to(
            config.biome_parameters,
            (batch, BIOME_CAPACITY, TERRAIN_PARAMETER_COUNT),
        ),
        column_known=known,
        terrain_height=jnp.where(known, fields.height, jnp.int16(0)),
        surface_cell_code=jnp.where(known, fields.surface, jnp.uint16(0)),
        subsurface_cell_code=jnp.where(
            known,
            fields.subsurface,
            jnp.uint16(0),
        ),
        biome_code=jnp.where(
            known,
            fields.biome_code,
            jnp.uint8(DEFAULT_BIOME_CODE),
        ),
        cave_span_mask=fields.cave_mask & known[..., None],
        cave_min_y=jnp.where(
            known[..., None],
            fields.cave_min_y,
            jnp.int16(0),
        ),
        cave_max_y=jnp.where(
            known[..., None],
            fields.cave_max_y,
            jnp.int16(0),
        ),
    )


def sample_surrogate_terrain_columns(
    seed_words: jax.Array,
    block_xz: jax.Array,
    config: TerrainGeneratorConfig,
) -> SurrogateTerrainColumnSample:
    """Sample arbitrary absolute columns without generating a dense tile."""

    seeds = jnp.asarray(seed_words, dtype=jnp.uint32)
    positions = jnp.asarray(block_xz, dtype=jnp.int32)
    if seeds.ndim != 2 or seeds.shape[1] != 2:
        raise ValueError("seed_words must have shape [batch, 2]")
    if (
        positions.ndim != 3
        or positions.shape[0] != seeds.shape[0]
        or positions.shape[2] != 2
    ):
        raise ValueError("block_xz must have shape [batch, queries, 2]")
    _validate_config_shapes(config)
    x = positions[..., 0]
    z = positions[..., 1]
    fields = _terrain_fields(seeds, x, z, config)
    return SurrogateTerrainColumnSample(
        valid=fields.valid,
        terrain_height=jnp.where(fields.valid, fields.height, jnp.int16(0)),
        surface_cell_code=jnp.where(
            fields.valid,
            fields.surface,
            jnp.uint16(0),
        ),
        subsurface_cell_code=jnp.where(
            fields.valid,
            fields.subsurface,
            jnp.uint16(0),
        ),
        biome_code=jnp.where(
            fields.valid,
            fields.biome_code,
            jnp.uint8(DEFAULT_BIOME_CODE),
        ),
    )


def _terrain_fields(
    seeds: jax.Array,
    x: jax.Array,
    z: jax.Array,
    config: TerrainGeneratorConfig,
) -> _TerrainFields:
    biome_field = jnp.float32(0.72) * _value_noise(
        seeds,
        x,
        z,
        _BIOME_LARGE_SCALE,
        0xB7E15162,
    ) + jnp.float32(0.28) * _value_noise(
        seeds,
        x,
        z,
        _BIOME_SMALL_SCALE,
        0x8AED2A6B,
    )
    match = (
        config.biome_mask
        & (biome_field[..., None] >= config.biome_minimum)
        & (biome_field[..., None] < config.biome_maximum)
    )
    has_biome = jnp.any(match, axis=-1)
    biome_index = jnp.argmax(match, axis=-1).astype(jnp.int32)
    selected = jnp.take(config.biome_parameters, biome_index, axis=0)
    selected = jnp.where(has_biome[..., None], selected, config.parameters)

    macro = _value_noise(seeds, x, z, _MACRO_SCALE, 0xA341316C)
    meso = _value_noise(seeds, x, z, _MESO_SCALE, 0xC8013EA4)
    detail = _value_noise(seeds, x, z, _DETAIL_SCALE, 0xAD90777D)
    ridge = 1.0 - jnp.abs(_value_noise(seeds, x, z, _RIDGE_SCALE, 0x7E95761E))
    raw_height = (
        selected[..., 0]
        + selected[..., 1] * macro
        + selected[..., 2] * meso
        + selected[..., 3] * detail
        + selected[..., 4] * ridge
    )
    valid = (
        _jax_parameters_valid(config)
        & jnp.isfinite(raw_height)
        & (raw_height >= jnp.float32(MIN_Y))
        & (raw_height < jnp.float32(MIN_Y + WORLD_HEIGHT))
    )
    height = jnp.floor(raw_height).astype(jnp.int16)
    surface = jnp.where(
        has_biome,
        jnp.take(config.surface_cell_code, biome_index),
        config.default_surface_cell_code,
    )
    subsurface = jnp.where(
        has_biome,
        jnp.take(config.subsurface_cell_code, biome_index),
        config.default_subsurface_cell_code,
    )
    biome_code = jnp.where(
        has_biome,
        biome_index,
        jnp.int32(DEFAULT_BIOME_CODE),
    ).astype(jnp.uint8)

    cave_masks = []
    cave_minimums = []
    cave_maximums = []
    previous_mask = jnp.zeros(height.shape, dtype=jnp.bool_)
    previous_maximum = jnp.zeros(height.shape, dtype=jnp.int16)
    for slot in range(CAVE_SPAN_CAPACITY):
        start = _CAVE_PARAMETER_START + slot * _CAVE_PARAMETERS_PER_SPAN
        floor_base = selected[..., start]
        span_height = selected[..., start + 1]
        threshold = selected[..., start + 2]
        floor_noise = _value_noise(
            seeds,
            x,
            z,
            _CAVE_FLOOR_SCALE[slot],
            0x51ED270B + slot * 0x10101,
        )
        height_noise = _value_noise(
            seeds,
            x,
            z,
            _CAVE_HEIGHT_SCALE[slot],
            0xD3A2646C + slot * 0x10101,
        )
        mask_noise = _value_noise(
            seeds,
            x,
            z,
            _CAVE_MASK_SCALE[slot],
            0x9E3779B9 + slot * 0x10101,
        )
        minimum = jnp.floor(floor_base + 4.0 * floor_noise).astype(jnp.int16)
        extent = jnp.maximum(
            jnp.int16(2),
            jnp.floor(span_height + 2.0 * height_noise).astype(jnp.int16),
        )
        maximum = minimum + extent
        enabled = (
            (span_height >= 2.0)
            & (mask_noise > threshold)
            & (minimum > MIN_Y)
            & (maximum <= height - 3)
            & (~previous_mask | (minimum >= previous_maximum + 2))
        )
        cave_masks.append(enabled)
        cave_minimums.append(jnp.where(enabled, minimum, jnp.int16(0)))
        cave_maximums.append(jnp.where(enabled, maximum, jnp.int16(0)))
        previous_mask = enabled
        previous_maximum = maximum
    return _TerrainFields(
        valid=valid,
        height=height,
        surface=surface.astype(jnp.uint16),
        subsurface=subsurface.astype(jnp.uint16),
        biome_code=biome_code,
        cave_mask=jnp.stack(cave_masks, axis=-1),
        cave_min_y=jnp.stack(cave_minimums, axis=-1),
        cave_max_y=jnp.stack(cave_maximums, axis=-1),
    )


def _jax_parameters_valid(config: TerrainGeneratorConfig) -> jax.Array:
    def vector_valid(values: jax.Array) -> jax.Array:
        spans = []
        for slot in range(CAVE_SPAN_CAPACITY):
            start = _CAVE_PARAMETER_START + slot * _CAVE_PARAMETERS_PER_SPAN
            floor = values[..., start]
            height = values[..., start + 1]
            threshold = values[..., start + 2]
            spans.append(
                (height == 0.0)
                | (
                    (height >= 2.0)
                    & (floor >= MIN_Y + 1)
                    & (floor + height < MIN_Y + WORLD_HEIGHT)
                    & (threshold >= -1.0)
                    & (threshold <= 1.0)
                )
            )
        return (
            jnp.all(jnp.isfinite(values), axis=-1)
            & jnp.all(values[..., 1:HEIGHT_PARAMETER_COUNT] >= 0.0, axis=-1)
            & jnp.all(
                values[..., SUPPORTED_TERRAIN_PARAMETERS:] == 0.0,
                axis=-1,
            )
            & jnp.all(jnp.stack(spans, axis=-1), axis=-1)
        )

    return vector_valid(config.parameters) & jnp.all(
        ~config.biome_mask | vector_valid(config.biome_parameters)
    )


def _value_noise(
    seeds: jax.Array,
    x: jax.Array,
    z: jax.Array,
    scale: int,
    salt: int,
) -> jax.Array:
    lattice_x = jnp.floor_divide(x, jnp.int32(scale))
    lattice_z = jnp.floor_divide(z, jnp.int32(scale))
    fraction_x = jnp.mod(x, jnp.int32(scale)).astype(jnp.float32) / scale
    fraction_z = jnp.mod(z, jnp.int32(scale)).astype(jnp.float32) / scale
    fraction_x = fraction_x * fraction_x * (3.0 - 2.0 * fraction_x)
    fraction_z = fraction_z * fraction_z * (3.0 - 2.0 * fraction_z)
    lower_left = _hash_noise(seeds, lattice_x, lattice_z, salt)
    lower_right = _hash_noise(seeds, lattice_x + 1, lattice_z, salt)
    upper_left = _hash_noise(seeds, lattice_x, lattice_z + 1, salt)
    upper_right = _hash_noise(
        seeds,
        lattice_x + 1,
        lattice_z + 1,
        salt,
    )
    lower = lower_left + fraction_x * (lower_right - lower_left)
    upper = upper_left + fraction_x * (upper_right - upper_left)
    return lower + fraction_z * (upper - lower)


def _hash_noise(
    seeds: jax.Array,
    x: jax.Array,
    z: jax.Array,
    salt: int,
) -> jax.Array:
    seed_shape = (seeds.shape[0],) + (1,) * (x.ndim - 1)
    seed_zero = jnp.reshape(seeds[:, 0], seed_shape)
    seed_one = jnp.reshape(seeds[:, 1], seed_shape)
    value = (
        x.astype(jnp.uint32) * jnp.uint32(0x9E3779B1)
        ^ z.astype(jnp.uint32) * jnp.uint32(0x85EBCA77)
        ^ seed_zero
        ^ (seed_one * jnp.uint32(0xC2B2AE3D))
        ^ jnp.uint32(salt)
    )
    value ^= value >> jnp.uint32(16)
    value *= jnp.uint32(0x7FEB352D)
    value ^= value >> jnp.uint32(15)
    value *= jnp.uint32(0x846CA68B)
    value ^= value >> jnp.uint32(16)
    unit = (value >> jnp.uint32(8)).astype(jnp.float32) * jnp.float32(1.0 / (1 << 24))
    return unit * 2.0 - 1.0


def _validate_config_shapes(config: TerrainGeneratorConfig) -> None:
    expected = {
        "parameters": (TERRAIN_PARAMETER_COUNT,),
        "biome_parameters": (
            BIOME_CAPACITY,
            TERRAIN_PARAMETER_COUNT,
        ),
        "biome_minimum": (BIOME_CAPACITY,),
        "biome_maximum": (BIOME_CAPACITY,),
        "biome_mask": (BIOME_CAPACITY,),
        "surface_cell_code": (BIOME_CAPACITY,),
        "subsurface_cell_code": (BIOME_CAPACITY,),
        "default_surface_cell_code": (),
        "default_subsurface_cell_code": (),
    }
    if not isinstance(config, TerrainGeneratorConfig):
        raise TypeError("config must be a TerrainGeneratorConfig")
    for name, shape in expected.items():
        if getattr(config, name).shape != shape:
            raise ValueError(f"terrain config {name} must have shape {shape}")


def _validate_parameter_vector(values: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must be finite")
    if np.any(values[1:HEIGHT_PARAMETER_COUNT] < 0.0):
        raise ValueError(f"{label} height amplitudes must be non-negative")
    if np.any(values[SUPPORTED_TERRAIN_PARAMETERS:] != 0.0):
        raise ValueError(f"{label} unsupported slots must remain zero")
    for slot in range(CAVE_SPAN_CAPACITY):
        start = _CAVE_PARAMETER_START + slot * _CAVE_PARAMETERS_PER_SPAN
        floor, height, threshold = values[start : start + 3]
        if height == 0.0:
            continue
        if (
            height < 2.0
            or floor < MIN_Y + 1
            or floor + height >= MIN_Y + WORLD_HEIGHT
            or not -1.0 <= threshold <= 1.0
        ):
            raise ValueError(f"{label} cave span {slot} is outside its schema")


def _cell_code(values: Mapping[str, int], biome: str, label: str) -> int:
    try:
        value = values[biome]
    except KeyError as error:
        raise ValueError(f"missing {label} cell code for {biome!r}") from error
    return _uint16(value, f"{label} cell code")


def _uint16(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error
    if result < 0 or result > np.iinfo(np.uint16).max:
        raise ValueError(f"{label} exceeds uint16")
    return result


__all__ = [
    "BIOME_CAPACITY",
    "CAVE_SPAN_CAPACITY",
    "DEFAULT_BIOME_CODE",
    "PRIMARY_ZONE_V1_TERRAIN_SOURCE",
    "PROFILE_ONLY_TERRAIN_SOURCE",
    "SURROGATE_TERRAIN_SOURCE_SCHEMA",
    "SurrogateTerrainTile",
    "SurrogateTerrainColumnSample",
    "SurrogateTerrainBiomeSource",
    "SurrogateTerrainSource",
    "TerrainGeneratorConfig",
    "TerrainGeneratorSpec",
    "default_terrain_parameters",
    "generate_surrogate_terrain",
    "plains_v1_biome_parameters",
    "primary_zone_v1_biome_parameters",
    "primary_zone_v1_terrain_generator_spec",
    "sample_surrogate_terrain_columns",
    "terrain_generator_spec_from_profile",
]
