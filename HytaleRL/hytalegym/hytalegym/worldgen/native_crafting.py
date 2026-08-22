"""Typed bridge contract for the runtime-resolved native crafting catalog."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping


NATIVE_CRAFTING_CATALOG_SCHEMA = (
    "hytalerl_native_crafting_catalog_evidence_v2"
)
NATIVE_CRAFTING_CATALOG_VERSION = 2
NATIVE_CRAFTING_RECIPE_CAPACITY = 1_947
NATIVE_CRAFTING_INPUT_CAPACITY = 28
NATIVE_CRAFTING_OUTPUT_CAPACITY = 4
NATIVE_CRAFTING_BENCH_CAPACITY = 3


@dataclass(frozen=True, slots=True)
class NativeCraftingMaterial:
    """One resolved native material selector or concrete output."""

    item_asset_id: str
    resource_type_id: str
    quantity: int
    metadata_json_sha256: str
    item_tag_present: bool
    excluded_item_asset_ids: tuple[str, ...]

    @classmethod
    def from_response(
        cls,
        value: Mapping[str, Any],
    ) -> NativeCraftingMaterial:
        item = _string(value, "item_asset_id", allow_empty=True)
        resource = _string(value, "resource_type_id", allow_empty=True)
        quantity = _integer(value, "quantity")
        metadata = _string(
            value,
            "metadata_json_sha256",
            allow_empty=True,
        )
        tagged = _boolean(value, "item_tag_present")
        exclusions = _string_tuple(value, "excluded_item_asset_ids")
        if (
            (not item and not resource and not tagged)
            or quantity < 1
            or (metadata and not _is_sha256(metadata))
            or exclusions != tuple(sorted(set(exclusions)))
        ):
            raise ValueError("native crafting material is outside capacity")
        return cls(
            item_asset_id=item,
            resource_type_id=resource,
            quantity=quantity,
            metadata_json_sha256=metadata,
            item_tag_present=tagged,
            excluded_item_asset_ids=exclusions,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "item_asset_id": self.item_asset_id,
            "resource_type_id": self.resource_type_id,
            "quantity": self.quantity,
            "metadata_json_sha256": self.metadata_json_sha256,
            "item_tag_present": self.item_tag_present,
            "excluded_item_asset_ids": list(self.excluded_item_asset_ids),
        }


@dataclass(frozen=True, slots=True)
class NativeCraftingBenchRequirement:
    """One native bench type/String-ID/tier predicate."""

    bench_type: int
    bench_id: str
    required_tier_level: int

    @classmethod
    def from_response(
        cls,
        value: Mapping[str, Any],
    ) -> NativeCraftingBenchRequirement:
        bench_type = _integer(value, "type")
        tier = _integer(value, "required_tier_level")
        if not 0 <= bench_type < 4 or tier < 0:
            raise ValueError("native crafting bench is outside its domain")
        return cls(
            bench_type=bench_type,
            bench_id=_string(value, "id"),
            required_tier_level=tier,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "type": self.bench_type,
            "id": self.bench_id,
            "required_tier_level": self.required_tier_level,
        }


@dataclass(frozen=True, slots=True)
class NativeCraftingRecipe:
    """One recipe after native inheritance and item generation."""

    recipe_id: str
    primary_output_item_asset_id: str
    inputs: tuple[NativeCraftingMaterial, ...]
    outputs: tuple[NativeCraftingMaterial, ...]
    bench_requirements: tuple[NativeCraftingBenchRequirement, ...]
    knowledge_required: bool
    required_memories_level: int
    time_seconds: float

    @classmethod
    def from_response(
        cls,
        value: Mapping[str, Any],
    ) -> NativeCraftingRecipe:
        inputs = _materials(value, "inputs")
        outputs = _materials(value, "outputs")
        raw_benches = _array(value, "bench_requirements")
        benches = tuple(
            NativeCraftingBenchRequirement.from_response(_mapping(item))
            for item in raw_benches
        )
        memories = _integer(value, "required_memories_level")
        time_seconds = _number(value, "time_seconds")
        if (
            not 1 <= len(inputs) <= NATIVE_CRAFTING_INPUT_CAPACITY
            or not 1 <= len(outputs) <= NATIVE_CRAFTING_OUTPUT_CAPACITY
            or len(benches) > NATIVE_CRAFTING_BENCH_CAPACITY
            or memories < 1
            or time_seconds < 0.0
        ):
            raise ValueError("native crafting recipe is outside capacity")
        return cls(
            recipe_id=_string(value, "recipe_id"),
            primary_output_item_asset_id=_string(
                value,
                "primary_output_item_asset_id",
                allow_empty=True,
            ),
            inputs=inputs,
            outputs=outputs,
            bench_requirements=benches,
            knowledge_required=_boolean(value, "knowledge_required"),
            required_memories_level=memories,
            time_seconds=time_seconds,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "recipe_id": self.recipe_id,
            "primary_output_item_asset_id": (
                self.primary_output_item_asset_id
            ),
            "inputs": [value.manifest() for value in self.inputs],
            "outputs": [value.manifest() for value in self.outputs],
            "bench_requirements": [
                value.manifest() for value in self.bench_requirements
            ],
            "knowledge_required": self.knowledge_required,
            "required_memories_level": self.required_memories_level,
            "time_seconds": self.time_seconds,
        }


@dataclass(frozen=True, slots=True)
class NativeCraftingCatalogEvidence:
    """Complete bridge-attributed catalog with explicit execution boundary."""

    bridge_sha256: str
    server_version: str
    world: str
    worldgen_provider: str
    worldgen_version: str
    seed: int
    recipes_without_bench_requirement: int
    tagged_material_count: int
    excluded_material_count: int
    recipes: tuple[NativeCraftingRecipe, ...]

    @classmethod
    def from_response(
        cls,
        value: Mapping[str, Any],
        *,
        expected_bridge_sha256: str | None = None,
    ) -> NativeCraftingCatalogEvidence:
        if value.get("type") != "crafting_catalog_evidence":
            raise ValueError("bridge returned the wrong crafting type")
        if value.get("schema") != NATIVE_CRAFTING_CATALOG_SCHEMA:
            raise ValueError("bridge returned the wrong crafting schema")
        if value.get("version") != NATIVE_CRAFTING_CATALOG_VERSION:
            raise ValueError("bridge returned the wrong crafting version")
        bridge = _sha256(_string(value, "bridge_sha256"), "bridge_sha256")
        if (
            expected_bridge_sha256 is not None
            and bridge.lower()
            != _sha256(
                expected_bridge_sha256,
                "expected_bridge_sha256",
            ).lower()
        ):
            raise ValueError("crafting catalog came from a different bridge")
        if (
            _integer(value, "expected_recipe_count")
            != NATIVE_CRAFTING_RECIPE_CAPACITY
            or _integer(value, "recipe_count")
            != NATIVE_CRAFTING_RECIPE_CAPACITY
            or value.get("native_recipe_identity")
            != "CraftRecipeAction.recipeId_string"
            or value.get("headless_npc_execution_supported") is not False
            or value.get("headless_npc_execution_boundary")
            != "CraftingManager.craftItem_requires_Player"
            or value.get(
                "headless_server_actor_execution_supported"
            ) is not True
            or value.get("headless_server_actor_execution_scope")
            != (
                "opt_in_Adventure_context_fieldcraft_and_Crafting_bench"
            )
            or value.get("headless_server_actor_execution_requires")
            != (
                "native_world_verbs_true_exact_inventory_and_recipe_context"
            )
            or value.get("public_player_execution_certified") is not False
        ):
            raise ValueError("native crafting capability boundary changed")
        recipes = tuple(
            NativeCraftingRecipe.from_response(_mapping(item))
            for item in _array(value, "recipes")
        )
        ids = tuple(recipe.recipe_id for recipe in recipes)
        if (
            len(recipes) != NATIVE_CRAFTING_RECIPE_CAPACITY
            or ids != tuple(sorted(set(ids)))
        ):
            raise ValueError("native crafting catalog is partial or unordered")
        missing_bench = sum(
            not recipe.bench_requirements for recipe in recipes
        )
        tagged = sum(
            material.item_tag_present
            for recipe in recipes
            for material in (*recipe.inputs, *recipe.outputs)
        )
        excluded = sum(
            bool(material.excluded_item_asset_ids)
            for recipe in recipes
            for material in recipe.inputs
        )
        reported = (
            _integer(value, "recipes_without_bench_requirement"),
            _integer(value, "tagged_material_count"),
            _integer(value, "excluded_material_count"),
        )
        if reported != (missing_bench, tagged, excluded):
            raise ValueError("native crafting catalog tallies are inconsistent")
        return cls(
            bridge_sha256=bridge,
            server_version=_string(value, "server_version"),
            world=_string(value, "world"),
            worldgen_provider=_string(value, "worldgen_provider"),
            worldgen_version=_string(value, "worldgen_version"),
            seed=_integer(value, "seed"),
            recipes_without_bench_requirement=missing_bench,
            tagged_material_count=tagged,
            excluded_material_count=excluded,
            recipes=recipes,
        )

    @property
    def packer_ready(self) -> bool:
        """Whether v1 can enter the current JAX packer without semantic loss."""

        return (
            self.recipes_without_bench_requirement == 0
            and self.tagged_material_count == 0
            and self.excluded_material_count == 0
        )

    def semantic_sha256(self) -> str:
        """Hash resolved meaning while excluding bridge/world provenance."""

        return _canonical_sha256({
            "schema": NATIVE_CRAFTING_CATALOG_SCHEMA,
            "version": NATIVE_CRAFTING_CATALOG_VERSION,
            "recipes": [recipe.manifest() for recipe in self.recipes],
        })


def native_crafting_catalog_evidence_request() -> dict[str, str]:
    """Build the bounded complete-catalog request."""

    return {"type": "crafting_catalog_evidence"}


def native_crafting_catalog_contract() -> dict[str, object]:
    """Return the versioned bridge-to-JAX catalog boundary."""

    return {
        "schema": NATIVE_CRAFTING_CATALOG_SCHEMA,
        "version": NATIVE_CRAFTING_CATALOG_VERSION,
        "server_version": "0.5.7",
        "identity": "CraftRecipeAction.recipeId_utf8_string",
        "fixed_capacity": {
            "recipes": NATIVE_CRAFTING_RECIPE_CAPACITY,
            "inputs_per_recipe": NATIVE_CRAFTING_INPUT_CAPACITY,
            "outputs_per_recipe": NATIVE_CRAFTING_OUTPUT_CAPACITY,
            "bench_requirements_per_recipe": (
                NATIVE_CRAFTING_BENCH_CAPACITY
            ),
        },
        "resolved_source": (
            "CraftingRecipe.getAssetMap_after_inheritance_and_item_generation"
        ),
        "material_fields": [
            "item_asset_id",
            "resource_type_id",
            "quantity",
            "metadata_json_sha256",
            "item_tag_present",
            "excluded_item_asset_ids",
        ],
        "fail_closed": [
            "partial_or_duplicate_catalog",
            "capacity_overflow",
            "bridge_identity_mismatch",
            "unresolved_bench_requirement",
            "unsupported_item_tag",
            "unrepresented_resource_output_exclusion",
        ],
        "execution": {
            "bare_headless_npc": "unsupported_requires_Player_component",
            "opt_in_headless_server_actor": (
                "native_CraftingManager_fieldcraft_and_ordinary_"
                "Crafting_bench"
            ),
            "opt_in_requirements": (
                "native_world_verbs_true_exact_inventory_recipe_and_"
                "bench_context"
            ),
            "specialized_benches": (
                "fail_closed_until_ordered_or_processing_inputs_are_"
                "transported"
            ),
            "public_player": "not_certified_until_deploy_and_live_differential",
            "catalog_export_is_not_action_acceptance": True,
        },
        "provenance": "native_runtime_bridge_plus_resolved_asset_map",
    }


def native_crafting_catalog_contract_sha256() -> str:
    return _canonical_sha256(native_crafting_catalog_contract())


def _materials(
    value: Mapping[str, Any],
    key: str,
) -> tuple[NativeCraftingMaterial, ...]:
    return tuple(
        NativeCraftingMaterial.from_response(_mapping(item))
        for item in _array(value, key)
    )


def _array(value: Mapping[str, Any], key: str) -> list[Any] | tuple[Any, ...]:
    result = value.get(key)
    if not isinstance(result, (list, tuple)):
        raise ValueError(f"{key} must be an array")
    return result


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("native crafting row must be an object")
    return value


def _string(
    value: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    result = value.get(key)
    if not isinstance(result, str) or (not allow_empty and not result):
        raise ValueError(f"{key} must be a string")
    return result


def _string_tuple(
    value: Mapping[str, Any],
    key: str,
) -> tuple[str, ...]:
    result = _array(value, key)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{key} must contain non-empty strings")
    return tuple(result)


def _integer(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int):
        raise ValueError(f"{key} must be an integer")
    return result


def _boolean(value: Mapping[str, Any], key: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise ValueError(f"{key} must be boolean")
    return result


def _number(value: Mapping[str, Any], key: str) -> float:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError(f"{key} must be numeric")
    number = float(result)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _sha256(value: str, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a SHA-256")
    return value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "NATIVE_CRAFTING_BENCH_CAPACITY",
    "NATIVE_CRAFTING_CATALOG_SCHEMA",
    "NATIVE_CRAFTING_CATALOG_VERSION",
    "NATIVE_CRAFTING_INPUT_CAPACITY",
    "NATIVE_CRAFTING_OUTPUT_CAPACITY",
    "NATIVE_CRAFTING_RECIPE_CAPACITY",
    "NativeCraftingBenchRequirement",
    "NativeCraftingCatalogEvidence",
    "NativeCraftingMaterial",
    "NativeCraftingRecipe",
    "native_crafting_catalog_contract",
    "native_crafting_catalog_contract_sha256",
    "native_crafting_catalog_evidence_request",
]
