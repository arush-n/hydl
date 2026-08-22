"""Source-bound surrogate placement for authored biome prefab families."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
import operator

from hytalegym.worldgen.surrogate.assets import (
    ASSIGNMENT_PREFIX,
    HytaleAssetArchive,
    HytaleAssetError,
)
from hytalegym.worldgen.surrogate.biomes import LocalBiomeCatalog
from hytalegym.worldgen.surrogate.contract import SurrogateWorldCapacity
from hytalegym.worldgen.surrogate.structures import (
    GeneratedStructurePlan,
    StructurePlacementConfig,
    SurrogateStructureCatalog,
    load_structure_catalog,
    plan_structure_tile,
)

SURROGATE_BIOME_PREFAB_LAYER_VERSION = 1
_ASSIGNMENT_ROOT_TYPES = frozenset({"Constant", "FieldFunction", "Weighted"})
_PROBABILITY_EPSILON = 1e-12


@dataclass(frozen=True)
class SurrogateBiomePrefabLayer:
    """One exact source family under an explicitly surrogate grid policy."""

    biome_id: str
    biome_code: int
    prop_graph_sha256: str
    prop_runtime: int
    assignment_id: str
    catalog: SurrogateStructureCatalog
    placement: StructurePlacementConfig

    def __post_init__(self) -> None:
        if not isinstance(self.biome_id, str) or not self.biome_id:
            raise TypeError("biome_id must be a non-empty string")
        code = _uint8(self.biome_code, "biome code")
        graph = _sha256(self.prop_graph_sha256, "prop graph SHA-256")
        runtime = _nonnegative_int(self.prop_runtime, "prop runtime")
        if (
            not isinstance(self.assignment_id, str)
            or not self.assignment_id
            or any(character in self.assignment_id for character in "/\\")
        ):
            raise TypeError("assignment_id must be a local non-empty string")
        if not isinstance(self.catalog, SurrogateStructureCatalog):
            raise TypeError("catalog must be a SurrogateStructureCatalog")
        if self.catalog.assignment_root_type not in _ASSIGNMENT_ROOT_TYPES:
            raise ValueError("unsupported source assignment root type")
        if not self.catalog.assignment_provenance.entry_path.endswith(
            f"/{self.assignment_id}.json"
        ):
            raise ValueError("assignment ID disagrees with catalog provenance")
        if not isinstance(self.placement, StructurePlacementConfig):
            raise TypeError("placement must be a StructurePlacementConfig")
        if self.placement.allowed_biome_codes != frozenset({code}):
            raise ValueError("placement biome mask disagrees with layer code")
        object.__setattr__(self, "biome_code", code)
        object.__setattr__(self, "prop_graph_sha256", graph)
        object.__setattr__(self, "prop_runtime", runtime)


def bind_surrogate_biome_prefab_layer(
    archive: HytaleAssetArchive,
    biome_catalog: LocalBiomeCatalog,
    *,
    biome_id: str,
    prop_graph_sha256: str,
    placement: StructurePlacementConfig,
) -> SurrogateBiomePrefabLayer:
    """Bind one active imported biome graph to exact prefab source assets."""

    if not isinstance(archive, HytaleAssetArchive):
        raise TypeError("archive must be a HytaleAssetArchive")
    if not isinstance(biome_catalog, LocalBiomeCatalog):
        raise TypeError("biome_catalog must be a LocalBiomeCatalog")
    if not isinstance(biome_id, str) or not biome_id:
        raise TypeError("biome_id must be a non-empty string")
    indexed = [
        (index, value)
        for index, value in enumerate(biome_catalog.biomes)
        if value.biome_id == biome_id
    ]
    if len(indexed) != 1:
        raise HytaleAssetError(f"biome ID resolved to {len(indexed)} catalog rows")
    code, biome = indexed[0]
    _uint8(code, "biome code")
    digest = _sha256(prop_graph_sha256, "prop graph SHA-256")
    if not isinstance(placement, StructurePlacementConfig):
        raise TypeError("placement must be a StructurePlacementConfig")

    matches = [
        prop for prop in biome.prop_assignments if prop.graph_sha256 == digest
    ]
    if len(matches) != 1:
        raise HytaleAssetError(
            f"biome prop graph resolved to {len(matches)} exact sources"
        )
    prop = matches[0]
    if prop.skipped:
        raise HytaleAssetError("skipped biome prop graph cannot be executed")
    if (
        prop.source_field != "Assignments"
        or prop.root_type != "Imported"
        or prop.assignment is None
        or prop.imported_assignments != (prop.assignment,)
    ):
        raise HytaleAssetError(
            "biome prop layer requires one imported Assignments root"
        )
    if prop.runtime is None:
        raise HytaleAssetError("biome prop layer requires an explicit runtime")
    assignment_id = prop.assignment
    if any(character in assignment_id for character in "/\\"):
        raise HytaleAssetError("biome prop assignment ID must be local")
    paths = archive.entry_paths(
        prefix=ASSIGNMENT_PREFIX,
        suffix=f"/{assignment_id}.json",
    )
    if len(paths) != 1:
        raise HytaleAssetError(
            f"biome prop assignment {assignment_id!r} resolved to {len(paths)} assets"
        )
    catalog = load_structure_catalog(
        archive,
        paths[0],
        accepted_root_types=_ASSIGNMENT_ROOT_TYPES,
    )
    allowed = placement.allowed_biome_codes
    expected = frozenset({code})
    if allowed is not None and allowed != expected:
        raise ValueError("placement biome codes disagree with the bound biome")
    return SurrogateBiomePrefabLayer(
        biome_id=biome.biome_id,
        biome_code=code,
        prop_graph_sha256=digest,
        prop_runtime=_nonnegative_int(prop.runtime, "prop runtime"),
        assignment_id=assignment_id,
        catalog=catalog,
        placement=replace(placement, allowed_biome_codes=expected),
    )


def plan_biome_prefab_tile(
    layers: Sequence[SurrogateBiomePrefabLayer],
    seed_words: tuple[int, int],
    *,
    tile_min_xz: tuple[int, int],
    column_sampler: object,
    capacity: SurrogateWorldCapacity | None = None,
) -> GeneratedStructurePlan:
    """Plan mutually exclusive biome layers on one shared global grid."""

    values = tuple(layers)
    if not values or any(
        not isinstance(value, SurrogateBiomePrefabLayer) for value in values
    ):
        raise TypeError("layers must contain SurrogateBiomePrefabLayer values")
    probability_windows = _ordered_layer_probability_windows(values)
    grid = {
        (
            value.placement.grid_spacing,
            value.placement.jitter,
            value.placement.salt,
        )
        for value in values
    }
    if len(grid) != 1:
        raise ValueError("biome prefab layers must share one placement grid")
    spacing, jitter, _salt = next(iter(grid))
    maximum_reach = max(value.catalog.maximum_horizontal_reach for value in values)
    if spacing - 2 * jitter <= 2 * maximum_reach:
        raise ValueError("shared biome prop grid cannot prove non-overlap")
    layout = SurrogateWorldCapacity() if capacity is None else capacity
    if not isinstance(layout, SurrogateWorldCapacity):
        raise TypeError("capacity must be a SurrogateWorldCapacity")

    plans = tuple(
        plan_structure_tile(
            layer.catalog,
            seed_words,
            tile_min_xz=tile_min_xz,
            column_sampler=column_sampler,
            config=layer.placement,
            capacity=layout,
            probability_start=probability_start,
        )
        for layer, probability_start in probability_windows
    )
    instances = [
        instance for plan in plans for instance in plan.instances
    ]
    cells = [instance.grid_cell for instance in instances]
    if len(set(cells)) != len(cells):
        raise HytaleAssetError("biome prop layers selected the same grid cell")
    if len(instances) > layout.structure_instance_capacity:
        raise HytaleAssetError(
            "biome prop plan exceeds instance capacity "
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
        tile_min_xz=plans[0].tile_min_xz,
        seed_words=plans[0].seed_words,
        instances=tuple(instances),
        considered_candidates=sum(plan.considered_candidates for plan in plans),
        probability_candidates=sum(
            plan.probability_candidates for plan in plans
        ),
        biome_rejections=sum(plan.biome_rejections for plan in plans),
        vertical_rejections=sum(plan.vertical_rejections for plan in plans),
        assignment_sha256=biome_prefab_source_bundle_sha256(values),
    )


def biome_prefab_layer_contract(
    layer: SurrogateBiomePrefabLayer,
) -> dict[str, object]:
    """Return the canonical non-native placement provenance for one layer."""

    if not isinstance(layer, SurrogateBiomePrefabLayer):
        raise TypeError("layer must be a SurrogateBiomePrefabLayer")
    source = layer.catalog.assignment_provenance
    placement = layer.placement
    return {
        "version": SURROGATE_BIOME_PREFAB_LAYER_VERSION,
        "mode": "source_bound_surrogate_grid_v1",
        "biome_id": layer.biome_id,
        "biome_code": layer.biome_code,
        "prop_graph_sha256": layer.prop_graph_sha256,
        "prop_runtime": layer.prop_runtime,
        "assignment_id": layer.assignment_id,
        "assignment_entry_path": source.entry_path,
        "assignment_sha256": source.content_sha256,
        "assignment_root_type": layer.catalog.assignment_root_type,
        "placement": {
            "grid_spacing": placement.grid_spacing,
            "jitter": placement.jitter,
            "placement_probability": placement.placement_probability,
            "salt": placement.salt,
            "allowed_biome_codes": sorted(placement.allowed_biome_codes or ()),
        },
        "native_distribution": False,
    }


def biome_prefab_source_bundle_sha256(
    layers: Sequence[SurrogateBiomePrefabLayer],
) -> str:
    source = tuple(layers)
    if not source or any(
        not isinstance(layer, SurrogateBiomePrefabLayer) for layer in source
    ):
        raise TypeError("layers must contain SurrogateBiomePrefabLayer values")
    values = sorted(
        (biome_prefab_layer_contract(layer) for layer in source),
        key=lambda value: (
            value["biome_code"],
            value["prop_graph_sha256"],
            value["assignment_id"],
        ),
    )
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _ordered_layer_probability_windows(
    layers: tuple[SurrogateBiomePrefabLayer, ...],
) -> tuple[tuple[SurrogateBiomePrefabLayer, float], ...]:
    ordered = tuple(
        sorted(
            layers,
            key=lambda value: (
                value.biome_code,
                value.prop_graph_sha256,
                value.assignment_id,
            ),
        )
    )
    code_by_name: dict[str, int] = {}
    name_by_code: dict[int, str] = {}
    seen_sources: set[tuple[int, str]] = set()
    totals: dict[int, float] = {}
    result: list[tuple[SurrogateBiomePrefabLayer, float]] = []
    for layer in ordered:
        prior_code = code_by_name.setdefault(layer.biome_id, layer.biome_code)
        prior_name = name_by_code.setdefault(layer.biome_code, layer.biome_id)
        if prior_code != layer.biome_code or prior_name != layer.biome_id:
            raise ValueError("biome prefab layer identity is ambiguous")
        source = (layer.biome_code, layer.prop_graph_sha256)
        if source in seen_sources:
            raise ValueError("biome prefab source is duplicated")
        seen_sources.add(source)
        start = totals.get(layer.biome_code, 0.0)
        total = math.fsum((start, layer.placement.placement_probability))
        if total > 1.0 + _PROBABILITY_EPSILON:
            raise ValueError("biome prefab placement probabilities exceed one")
        result.append((layer, start))
        totals[layer.biome_code] = min(total, 1.0)
    return tuple(result)


def _sha256(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    result = value.lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{label} must contain 64 hexadecimal digits")
    return result


def _uint8(value: int, label: str) -> int:
    result = _integer(value, label)
    if not 0 <= result <= 255:
        raise ValueError(f"{label} exceeds uint8")
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
    "SURROGATE_BIOME_PREFAB_LAYER_VERSION",
    "SurrogateBiomePrefabLayer",
    "bind_surrogate_biome_prefab_layer",
    "biome_prefab_layer_contract",
    "biome_prefab_source_bundle_sha256",
    "plan_biome_prefab_tile",
]
