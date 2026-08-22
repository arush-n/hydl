"""Coordinate-keyed offline structure catalogs and placement plans."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import operator

import numpy as np

from hytalegym.worldgen.region import (
    CAPTURE_BLOCKS_PER_AXIS,
    CHUNK_SIZE,
    MIN_Y,
    WORLD_HEIGHT,
)
from hytalegym.worldgen.surrogate.assets import (
    PREFAB_PREFIX,
    AssetProvenance,
    AssignmentPrefabReference,
    HytaleAssetArchive,
    HytaleAssetError,
    PrefabTemplate,
)
from hytalegym.worldgen.surrogate.contract import SurrogateWorldCapacity
from hytalegym.worldgen.surrogate.prefab import (
    CompiledPrefabOverlay,
    PrefabCompilationPolicy,
    PrefabPlacement,
    compile_prefab_overlay,
    place_prefab_entity_transform,
)

ColumnSampler = Callable[
    [np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]
_UINT64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class StructurePrefabFamily:
    reference: AssignmentPrefabReference
    templates: tuple[PrefabTemplate, ...]


@dataclass(frozen=True)
class SurrogateStructureCatalog:
    assignment_provenance: AssetProvenance
    assignment_root_type: str
    families: tuple[StructurePrefabFamily, ...]
    maximum_horizontal_reach: int

    @property
    def template_count(self) -> int:
        return sum(len(family.templates) for family in self.families)


@dataclass(frozen=True)
class StructurePlacementConfig:
    """Calibratable, explicitly surrogate placement distribution."""

    grid_spacing: int = 48
    jitter: int = 4
    placement_probability: float = 0.08
    salt: int = 0x5354525543545552
    allowed_biome_codes: frozenset[int] | None = None

    def __post_init__(self) -> None:
        spacing = _positive_int(self.grid_spacing, "grid_spacing")
        jitter = _nonnegative_int(self.jitter, "jitter")
        if 2 * jitter >= spacing:
            raise ValueError("structure jitter must be less than half the spacing")
        probability = float(self.placement_probability)
        if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise ValueError("placement_probability must be in [0, 1]")
        salt = _integer(self.salt, "salt") & _UINT64_MASK
        allowed = self.allowed_biome_codes
        if allowed is not None:
            if not isinstance(allowed, frozenset):
                raise TypeError("allowed_biome_codes must be a frozenset")
            normalized = frozenset(
                _uint8(value, "allowed biome code") for value in allowed
            )
            object.__setattr__(self, "allowed_biome_codes", normalized)
        object.__setattr__(self, "grid_spacing", spacing)
        object.__setattr__(self, "jitter", jitter)
        object.__setattr__(self, "placement_probability", probability)
        object.__setattr__(self, "salt", salt)


@dataclass(frozen=True)
class GeneratedStructureInstance:
    grid_cell: tuple[int, int]
    random_key: int
    family_path: str
    prefab_entry_path: str
    anchor_world: tuple[int, int, int]
    quarter_turns: int
    load_entities: bool
    placement: PrefabPlacement


@dataclass(frozen=True)
class GeneratedStructurePlan:
    tile_min_xz: tuple[int, int]
    seed_words: tuple[int, int]
    instances: tuple[GeneratedStructureInstance, ...]
    considered_candidates: int
    probability_candidates: int
    biome_rejections: int
    vertical_rejections: int
    assignment_sha256: str

    @property
    def placements(self) -> tuple[PrefabPlacement, ...]:
        return tuple(instance.placement for instance in self.instances)


def load_structure_catalog(
    archive: HytaleAssetArchive,
    assignment_entry_path: str,
    *,
    accepted_root_types: frozenset[str] = frozenset({"Weighted"}),
    family_template_capacity: int = 256,
    total_template_capacity: int = 2_048,
) -> SurrogateStructureCatalog:
    """Expand weighted local prefab directories without copying assets."""

    if not isinstance(archive, HytaleAssetArchive):
        raise TypeError("archive must be a HytaleAssetArchive")
    per_family = _positive_int(
        family_template_capacity,
        "family template capacity",
    )
    total_capacity = _positive_int(
        total_template_capacity,
        "total template capacity",
    )
    if (
        not isinstance(accepted_root_types, frozenset)
        or not accepted_root_types
        or any(not isinstance(value, str) or not value for value in accepted_root_types)
    ):
        raise TypeError("accepted_root_types must be a non-empty frozenset of strings")
    assignment = archive.load_assignment_index(assignment_entry_path)
    if assignment.root_type not in accepted_root_types:
        raise HytaleAssetError(
            f"unsupported structure assignment type {assignment.root_type!r}"
        )
    if not assignment.prefab_references:
        raise HytaleAssetError("structure assignment contains no prefab paths")

    families: list[StructurePrefabFamily] = []
    seen_paths: set[str] = set()
    maximum_reach = 0
    total = 0
    for reference in assignment.prefab_references:
        prefix = f"{PREFAB_PREFIX}{reference.path.strip('/')}"
        matches = tuple(
            path
            for path in archive.entry_paths(
                prefix=prefix,
                suffix=".prefab.json",
            )
            if path == f"{prefix}.prefab.json" or path.startswith(f"{prefix}/")
        )
        if not matches:
            raise HytaleAssetError(
                f"prefab family {reference.path!r} contains no prefabs"
            )
        if len(matches) > per_family:
            raise HytaleAssetError(
                f"prefab family exceeds template capacity {per_family}"
            )
        if seen_paths.intersection(matches):
            raise HytaleAssetError("prefab families overlap")
        seen_paths.update(matches)
        total += len(matches)
        if total > total_capacity:
            raise HytaleAssetError(
                f"structure catalog exceeds template capacity {total_capacity}"
            )
        templates = tuple(archive.load_prefab(path) for path in matches)
        for template in templates:
            maximum_reach = max(
                maximum_reach,
                _template_horizontal_reach(
                    template,
                    include_entities=reference.load_entities,
                ),
            )
        families.append(
            StructurePrefabFamily(
                reference=reference,
                templates=templates,
            )
        )
    return SurrogateStructureCatalog(
        assignment_provenance=assignment.provenance,
        assignment_root_type=assignment.root_type,
        families=tuple(families),
        maximum_horizontal_reach=maximum_reach,
    )


def plan_structure_tile(
    catalog: SurrogateStructureCatalog,
    seed_words: tuple[int, int],
    *,
    tile_min_xz: tuple[int, int],
    column_sampler: ColumnSampler,
    config: StructurePlacementConfig | None = None,
    capacity: SurrogateWorldCapacity | None = None,
    probability_start: float = 0.0,
) -> GeneratedStructurePlan:
    """Plan one probability interval of global-grid prefabs intersecting a tile."""

    if not isinstance(catalog, SurrogateStructureCatalog):
        raise TypeError("catalog must be a SurrogateStructureCatalog")
    seeds = tuple(_uint32(value, "seed word") for value in seed_words)
    if len(seeds) != 2:
        raise ValueError("seed_words must contain two uint32 values")
    tile_min = _int32_pair(tile_min_xz, "tile_min_xz")
    if any(value % CHUNK_SIZE for value in tile_min):
        raise ValueError("tile_min_xz must be chunk aligned")
    if not callable(column_sampler):
        raise TypeError("column_sampler must be callable")
    settings = StructurePlacementConfig() if config is None else config
    if not isinstance(settings, StructurePlacementConfig):
        raise TypeError("config must be a StructurePlacementConfig")
    probability_min, probability_max = _probability_interval(
        probability_start,
        settings.placement_probability,
    )
    layout = SurrogateWorldCapacity() if capacity is None else capacity
    if not isinstance(layout, SurrogateWorldCapacity):
        raise TypeError("capacity must be a SurrogateWorldCapacity")
    if (
        settings.grid_spacing - 2 * settings.jitter
        <= 2 * catalog.maximum_horizontal_reach
    ):
        raise ValueError("structure spacing cannot prove non-overlapping prefab bounds")

    candidates = _candidate_records(
        catalog,
        seeds,
        tile_min,
        settings,
    )
    probability_candidates = [
        candidate
        for candidate in candidates
        if probability_min <= _unit_float(candidate[2]) < probability_max
    ]
    positions = np.asarray(
        [(candidate[3], candidate[4]) for candidate in probability_candidates],
        dtype=np.int32,
    ).reshape(-1, 2)
    valid, heights, biomes = column_sampler(positions)
    valid = np.asarray(valid, dtype=np.bool_)
    heights = np.asarray(heights)
    biomes = np.asarray(biomes)
    expected = (len(probability_candidates),)
    if valid.shape != expected or heights.shape != expected or biomes.shape != expected:
        raise ValueError("column_sampler returned invalid shapes")
    if np.any(valid & ((heights < MIN_Y) | (heights >= MIN_Y + WORLD_HEIGHT))):
        raise ValueError("column_sampler returned an out-of-world height")

    instances: list[GeneratedStructureInstance] = []
    biome_rejections = 0
    vertical_rejections = 0
    for candidate, is_valid, height, biome in zip(
        probability_candidates,
        valid,
        heights,
        biomes,
        strict=True,
    ):
        if not is_valid:
            raise HytaleAssetError("structure anchor terrain is unavailable")
        if (
            settings.allowed_biome_codes is not None
            and _uint8(biome, "sampled biome code") not in settings.allowed_biome_codes
        ):
            biome_rejections += 1
            continue
        (
            grid_cell,
            family,
            random_key,
            anchor_x,
            anchor_z,
            template,
            turns,
        ) = candidate
        anchor = (anchor_x, int(height), anchor_z)
        bounds_min, bounds_max = _placed_bounds(
            template,
            anchor,
            turns,
            include_entities=family.reference.load_entities,
        )
        if bounds_min[1] < MIN_Y or bounds_max[1] >= MIN_Y + WORLD_HEIGHT:
            vertical_rejections += 1
            continue
        if not _intersects_tile(bounds_min, bounds_max, tile_min):
            continue
        instances.append(
            GeneratedStructureInstance(
                grid_cell=grid_cell,
                random_key=random_key,
                family_path=family.reference.path,
                prefab_entry_path=template.provenance.entry_path,
                anchor_world=anchor,
                quarter_turns=turns,
                load_entities=family.reference.load_entities,
                placement=PrefabPlacement(template, anchor, turns),
            )
        )
        if len(instances) > layout.structure_instance_capacity:
            raise HytaleAssetError(
                "structure plan exceeds instance capacity "
                f"{layout.structure_instance_capacity}"
            )
    instances.sort(
        key=lambda value: (
            value.anchor_world,
            value.prefab_entry_path,
            value.quarter_turns,
        )
    )
    return GeneratedStructurePlan(
        tile_min_xz=tile_min,
        seed_words=seeds,
        instances=tuple(instances),
        considered_candidates=len(candidates),
        probability_candidates=len(probability_candidates),
        biome_rejections=biome_rejections,
        vertical_rejections=vertical_rejections,
        assignment_sha256=catalog.assignment_provenance.content_sha256,
    )


def _probability_interval(start: float, width: float) -> tuple[float, float]:
    if (
        isinstance(start, bool)
        or isinstance(width, bool)
        or not isinstance(start, (int, float))
        or not isinstance(width, (int, float))
    ):
        raise TypeError("structure probability interval must be numeric")
    minimum = float(start)
    maximum = math.fsum((minimum, float(width)))
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum < 0.0
        or maximum > 1.0 + 1e-12
    ):
        raise ValueError("structure probability interval must lie in [0, 1]")
    return minimum, min(maximum, 1.0)


def compile_structure_plan(
    plan: GeneratedStructurePlan,
    *,
    capacity: SurrogateWorldCapacity | None = None,
    ignored_component_types: frozenset[str] = frozenset(),
) -> CompiledPrefabOverlay | None:
    """Compile a plan, allowing only expected outer-halo clipping."""

    if not isinstance(plan, GeneratedStructurePlan):
        raise TypeError("plan must be a GeneratedStructurePlan")
    if not plan.instances:
        return None
    layout = SurrogateWorldCapacity() if capacity is None else capacity
    if not isinstance(layout, SurrogateWorldCapacity):
        raise TypeError("capacity must be a SurrogateWorldCapacity")
    policy = PrefabCompilationPolicy(
        ignored_component_types=ignored_component_types,
        allow_clipping=True,
    )
    block_placements = tuple(
        placement for placement in plan.placements if placement.template.blocks
    )
    if not block_placements:
        return None
    return compile_prefab_overlay(
        block_placements,
        tile_min_xz=plan.tile_min_xz,
        capacity=layout.structure_cell_capacity,
        policy=policy,
    )


def _candidate_records(
    catalog: SurrogateStructureCatalog,
    seeds: tuple[int, int],
    tile_min: tuple[int, int],
    settings: StructurePlacementConfig,
) -> list[
    tuple[
        tuple[int, int],
        StructurePrefabFamily,
        int,
        int,
        int,
        PrefabTemplate,
        int,
    ]
]:
    reach = catalog.maximum_horizontal_reach
    spacing = settings.grid_spacing
    center = spacing // 2
    minimum = (
        tile_min[0] - reach - center - settings.jitter,
        tile_min[1] - reach - center - settings.jitter,
    )
    maximum = (
        tile_min[0] + CAPTURE_BLOCKS_PER_AXIS + reach - center + settings.jitter,
        tile_min[1] + CAPTURE_BLOCKS_PER_AXIS + reach - center + settings.jitter,
    )
    cell_min = (
        minimum[0] // spacing - 1,
        minimum[1] // spacing - 1,
    )
    cell_max = (
        maximum[0] // spacing + 1,
        maximum[1] // spacing + 1,
    )
    weights = [family.reference.weight for family in catalog.families]
    weight_sum = sum(weights)
    result = []
    for cell_x in range(cell_min[0], cell_max[0] + 1):
        for cell_z in range(cell_min[1], cell_max[1] + 1):
            random_key = _coordinate_hash(
                seeds,
                cell_x,
                cell_z,
                settings.salt,
            )
            family = _weighted_family(
                catalog.families,
                weights,
                weight_sum,
                _coordinate_hash(seeds, cell_x, cell_z, random_key ^ 0x11),
            )
            template_key = _coordinate_hash(
                seeds,
                cell_x,
                cell_z,
                random_key ^ 0x22,
            )
            template = family.templates[template_key % len(family.templates)]
            turns = int(
                _coordinate_hash(
                    seeds,
                    cell_x,
                    cell_z,
                    random_key ^ 0x33,
                )
                & 3
            )
            jitter_span = 2 * settings.jitter + 1
            jitter_x = (
                _coordinate_hash(
                    seeds,
                    cell_x,
                    cell_z,
                    random_key ^ 0x44,
                )
                % jitter_span
                - settings.jitter
            )
            jitter_z = (
                _coordinate_hash(
                    seeds,
                    cell_x,
                    cell_z,
                    random_key ^ 0x55,
                )
                % jitter_span
                - settings.jitter
            )
            anchor_x = cell_x * spacing + center + jitter_x
            anchor_z = cell_z * spacing + center + jitter_z
            _int32(anchor_x, "structure anchor X")
            _int32(anchor_z, "structure anchor Z")
            if not template.blocks and not (
                family.reference.load_entities and template.entities
            ):
                continue
            xz_min, xz_max = _placed_xz_bounds(
                template,
                anchor_x,
                anchor_z,
                turns,
                include_entities=family.reference.load_entities,
            )
            if (
                xz_max[0] < tile_min[0]
                or xz_min[0] >= tile_min[0] + CAPTURE_BLOCKS_PER_AXIS
                or xz_max[1] < tile_min[1]
                or xz_min[1] >= tile_min[1] + CAPTURE_BLOCKS_PER_AXIS
            ):
                continue
            result.append(
                (
                    (cell_x, cell_z),
                    family,
                    random_key,
                    anchor_x,
                    anchor_z,
                    template,
                    turns,
                )
            )
    return result


def _weighted_family(
    families: tuple[StructurePrefabFamily, ...],
    weights: list[float],
    total: float,
    random_key: int,
) -> StructurePrefabFamily:
    target = _unit_float(random_key) * total
    cumulative = 0.0
    for family, weight in zip(families, weights, strict=True):
        cumulative += weight
        if target < cumulative:
            return family
    return families[-1]


def _coordinate_hash(
    seeds: tuple[int, int],
    x: int,
    z: int,
    salt: int,
) -> int:
    value = (
        seeds[0]
        ^ (seeds[1] << 32)
        ^ ((x & _UINT64_MASK) * 0x9E3779B185EBCA87)
        ^ ((z & _UINT64_MASK) * 0xC2B2AE3D27D4EB4F)
        ^ salt
    ) & _UINT64_MASK
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _UINT64_MASK
    value ^= value >> 31
    return value


def _unit_float(value: int) -> float:
    return (value & _UINT64_MASK) / float(1 << 64)


def _template_horizontal_reach(
    template: PrefabTemplate,
    *,
    include_entities: bool,
) -> int:
    offsets = [
        (
            block.position[0] - template.anchor[0],
            block.position[2] - template.anchor[2],
        )
        for block in template.blocks
    ]
    if include_entities:
        for entity in template.entities:
            for turns in range(4):
                position, _rotation = place_prefab_entity_transform(
                    entity,
                    template.anchor,
                    (0, 0, 0),
                    turns,
                )
                offsets.append((math.floor(position[0]), math.floor(position[2])))
    return max((max(abs(x), abs(z)) for x, z in offsets), default=0)


def _placed_bounds(
    template: PrefabTemplate,
    anchor: tuple[int, int, int],
    turns: int,
    *,
    include_entities: bool,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    positions = [
        tuple(anchor[axis] + rotated[axis] for axis in range(3))
        for block in template.blocks
        for rotated in (
            _rotate_y(
                tuple(
                    block.position[axis] - template.anchor[axis] for axis in range(3)
                ),
                turns,
            ),
        )
    ]
    if include_entities:
        positions.extend(
            tuple(math.floor(value) for value in position)
            for entity in template.entities
            for position, _rotation in (
                place_prefab_entity_transform(
                    entity,
                    template.anchor,
                    anchor,
                    turns,
                ),
            )
        )
    if not positions:
        raise HytaleAssetError("prefab placement has no enabled content")
    return (
        tuple(min(value[axis] for value in positions) for axis in range(3)),
        tuple(max(value[axis] for value in positions) for axis in range(3)),
    )


def _placed_xz_bounds(
    template: PrefabTemplate,
    anchor_x: int,
    anchor_z: int,
    turns: int,
    *,
    include_entities: bool,
) -> tuple[tuple[int, int], tuple[int, int]]:
    minimum, maximum = _placed_bounds(
        template,
        (anchor_x, 0, anchor_z),
        turns,
        include_entities=include_entities,
    )
    return (minimum[0], minimum[2]), (maximum[0], maximum[2])


def _intersects_tile(
    minimum: tuple[int, int, int],
    maximum: tuple[int, int, int],
    tile_min: tuple[int, int],
) -> bool:
    return (
        maximum[0] >= tile_min[0]
        and minimum[0] < tile_min[0] + CAPTURE_BLOCKS_PER_AXIS
        and maximum[2] >= tile_min[1]
        and minimum[2] < tile_min[1] + CAPTURE_BLOCKS_PER_AXIS
    )


def _rotate_y(
    value: tuple[int, int, int],
    turns: int,
) -> tuple[int, int, int]:
    x, y, z = value
    return (
        (x, y, z),
        (z, y, -x),
        (-x, y, -z),
        (-z, y, x),
    )[turns]


def _int32_pair(
    value: tuple[int, int],
    label: str,
) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{label} must be a 2-tuple")
    return _int32(value[0], label), _int32(value[1], label)


def _uint8(value: int, label: str) -> int:
    result = _integer(value, label)
    if not 0 <= result <= np.iinfo(np.uint8).max:
        raise ValueError(f"{label} exceeds uint8")
    return result


def _uint32(value: int, label: str) -> int:
    result = _integer(value, label)
    if not 0 <= result <= np.iinfo(np.uint32).max:
        raise ValueError(f"{label} exceeds uint32")
    return result


def _int32(value: int, label: str) -> int:
    result = _integer(value, label)
    if not np.iinfo(np.int32).min <= result <= np.iinfo(np.int32).max:
        raise ValueError(f"{label} exceeds int32")
    return result


def _positive_int(value: int, label: str) -> int:
    result = _integer(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_int(value: int, label: str) -> int:
    result = _integer(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _integer(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error


__all__ = [
    "GeneratedStructureInstance",
    "GeneratedStructurePlan",
    "StructurePlacementConfig",
    "StructurePrefabFamily",
    "SurrogateStructureCatalog",
    "compile_structure_plan",
    "load_structure_catalog",
    "plan_structure_tile",
]
