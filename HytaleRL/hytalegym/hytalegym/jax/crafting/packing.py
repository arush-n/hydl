"""Deterministic host packing for server-resolved crafting recipes.

This module deliberately does not decode raw Hytale asset JSON. Item recipe
generation and asset inheritance are server responsibilities; callers supply
the resolved recipe rows and the semantic-ID functions used by their inventory
domain. The resulting table keeps the native String recipe IDs alongside the
JAX arrays so a bridge never mistakes a table-local index for a native ID.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real

import jax.numpy as jnp
import numpy as np

from hytalegym.jax.crafting.contract import (
    ABSENT_ID,
    BENCH_TYPE_COUNT,
    IDENTITY_HASH_WORDS,
    MAX_BENCH_REQUIREMENTS,
    MAX_INGREDIENTS,
    MAX_OUTPUTS,
    METADATA_HASH_WORDS,
    RECIPE_TABLE_CAPACITY,
    crafting_contract_sha256,
    identity_sha256_words,
)
from hytalegym.jax.crafting.runtime import validate_recipe_table
from hytalegym.jax.crafting.types import RecipeTable


SemanticIdResolver = Callable[[str], int]
_RESOLVED_RECIPE_TABLE_SCHEMA = "hytalerl_resolved_recipe_table_v1"
_INT32_MAX = np.iinfo(np.int32).max


@dataclass(frozen=True, slots=True)
class ResolvedMaterial:
    """One server-resolved recipe material.

    Inputs may name an item, a resource type, or both. Outputs must name an
    item because the JAX runtime inserts concrete ``ItemStack`` values.
    """

    quantity: int
    item_asset_id: str | None = None
    resource_type_id: str | None = None
    metadata_hash: tuple[int, ...] = (0,) * METADATA_HASH_WORDS


@dataclass(frozen=True, slots=True)
class ResolvedBenchRequirement:
    """One resolved native ``BenchRequirement``."""

    bench_type: int
    bench_id: str
    required_tier_level: int = 0


@dataclass(frozen=True, slots=True)
class ResolvedRecipe:
    """One recipe after native asset inheritance and item generation."""

    recipe_id: str
    inputs: tuple[ResolvedMaterial, ...]
    outputs: tuple[ResolvedMaterial, ...]
    bench_requirements: tuple[ResolvedBenchRequirement, ...]
    knowledge_required: bool = False
    required_memories_level: int = 1
    time_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class PackedRecipeTable:
    """JAX table plus the host-only native identity mapping and provenance."""

    table: RecipeTable
    recipe_ids: tuple[str, ...]
    source_sha256: str
    content_sha256: str


def pack_resolved_recipe_table(
    recipes: Sequence[ResolvedRecipe],
    *,
    source_sha256: str,
    expected_recipe_count: int,
    item_id_resolver: SemanticIdResolver,
    resource_type_id_resolver: SemanticIdResolver,
) -> PackedRecipeTable:
    """Pack resolved recipes in canonical String-ID order.

    ``source_sha256`` identifies the asset tree or native export that produced
    the rows. ``expected_recipe_count`` makes partial exports fail closed.
    Semantic ID resolvers are explicit so this standalone package cannot
    silently choose a different integer domain from Combat inventory.
    """

    source_identity = _sha256(source_sha256, "source_sha256")
    if not callable(item_id_resolver):
        raise TypeError("item_id_resolver must be callable")
    if not callable(resource_type_id_resolver):
        raise TypeError("resource_type_id_resolver must be callable")

    rows = tuple(recipes)
    expected_count = _exact_integer(
        expected_recipe_count,
        "expected_recipe_count",
        minimum=0,
        maximum=RECIPE_TABLE_CAPACITY,
    )
    if any(not isinstance(recipe, ResolvedRecipe) for recipe in rows):
        raise TypeError("recipes must contain ResolvedRecipe values")
    if len(rows) > RECIPE_TABLE_CAPACITY:
        raise ValueError(
            f"resolved recipe count {len(rows)} exceeds capacity "
            f"{RECIPE_TABLE_CAPACITY}"
        )
    if len(rows) != expected_count:
        raise ValueError(
            f"resolved recipe count {len(rows)} does not match "
            f"expected_recipe_count {expected_count}"
        )
    recipe_ids = tuple(recipe.recipe_id for recipe in rows)
    if any(not isinstance(recipe_id, str) or not recipe_id for recipe_id in recipe_ids):
        raise ValueError("every resolved recipe needs a non-empty String ID")
    if len(set(recipe_ids)) != len(recipe_ids):
        raise ValueError("resolved recipe IDs must be unique")
    rows = tuple(sorted(rows, key=lambda recipe: recipe.recipe_id))
    recipe_ids = tuple(recipe.recipe_id for recipe in rows)

    arrays = _empty_table_arrays()
    manifest_rows: list[dict[str, object]] = []
    for index, recipe in enumerate(rows):
        manifest_rows.append(
            _pack_recipe(
                arrays,
                index,
                recipe,
                item_id_resolver=item_id_resolver,
                resource_type_id_resolver=resource_type_id_resolver,
            )
        )
    _validate_semantic_identity_mapping(manifest_rows)

    host_table = RecipeTable(**arrays)
    validate_recipe_table(host_table)
    table = RecipeTable(**{name: jnp.asarray(value) for name, value in arrays.items()})
    manifest = {
        "schema": _RESOLVED_RECIPE_TABLE_SCHEMA,
        "crafting_contract_sha256": crafting_contract_sha256(),
        "source_sha256": source_identity,
        "expected_recipe_count": expected_count,
        "recipes": manifest_rows,
    }
    content = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return PackedRecipeTable(
        table=table,
        recipe_ids=recipe_ids,
        source_sha256=source_identity,
        content_sha256=hashlib.sha256(content).hexdigest().upper(),
    )


def _pack_recipe(
    arrays: dict[str, np.ndarray],
    index: int,
    recipe: ResolvedRecipe,
    *,
    item_id_resolver: SemanticIdResolver,
    resource_type_id_resolver: SemanticIdResolver,
) -> dict[str, object]:
    if not isinstance(recipe, ResolvedRecipe):
        raise TypeError("recipes must contain ResolvedRecipe values")
    if len(recipe.inputs) > MAX_INGREDIENTS:
        raise ValueError(
            f"recipe {recipe.recipe_id!r} exceeds {MAX_INGREDIENTS} inputs"
        )
    if not recipe.outputs or len(recipe.outputs) > MAX_OUTPUTS:
        raise ValueError(
            f"recipe {recipe.recipe_id!r} must have 1..{MAX_OUTPUTS} outputs"
        )
    if not recipe.bench_requirements or (
        len(recipe.bench_requirements) > MAX_BENCH_REQUIREMENTS
    ):
        raise ValueError(
            f"recipe {recipe.recipe_id!r} must have "
            f"1..{MAX_BENCH_REQUIREMENTS} resolved bench requirements"
        )
    memories_level = _exact_integer(
        recipe.required_memories_level,
        "required_memories_level",
        minimum=1,
        maximum=_INT32_MAX,
    )
    time_seconds = _float32(
        recipe.time_seconds,
        "time_seconds",
        minimum=0.0,
    )
    if not isinstance(recipe.knowledge_required, (bool, np.bool_)):
        raise TypeError("knowledge_required must be boolean")

    arrays["recipe_mask"][index] = True
    arrays["recipe_id_hash"][index] = identity_sha256_words(recipe.recipe_id)
    inputs = [
        _material_manifest(
            material,
            item_id_resolver=item_id_resolver,
            resource_type_id_resolver=resource_type_id_resolver,
            output=False,
        )
        for material in recipe.inputs
    ]
    outputs = [
        _material_manifest(
            material,
            item_id_resolver=item_id_resolver,
            resource_type_id_resolver=resource_type_id_resolver,
            output=True,
        )
        for material in recipe.outputs
    ]
    requirements = [
        _requirement_manifest(requirement) for requirement in recipe.bench_requirements
    ]

    for slot, material in enumerate(inputs):
        arrays["input_mask"][index, slot] = True
        arrays["input_item_id"][index, slot] = material["item_id"]
        arrays["input_resource_type_id"][index, slot] = material["resource_type_id"]
        arrays["input_quantity"][index, slot] = material["quantity"]
        metadata = material["metadata_hash"]
        arrays["input_metadata_hash"][index, slot] = metadata
        arrays["input_metadata_required"][index, slot] = any(metadata)

    for slot, material in enumerate(outputs):
        arrays["output_mask"][index, slot] = True
        arrays["output_item_id"][index, slot] = material["item_id"]
        arrays["output_quantity"][index, slot] = material["quantity"]
        arrays["output_metadata_hash"][index, slot] = material["metadata_hash"]

    for slot, requirement in enumerate(requirements):
        arrays["requirement_mask"][index, slot] = True
        arrays["requirement_bench_type"][index, slot] = requirement["bench_type"]
        arrays["requirement_bench_id_hash"][index, slot] = identity_sha256_words(
            requirement["bench_id"]
        )
        arrays["requirement_tier_level"][index, slot] = requirement[
            "required_tier_level"
        ]

    arrays["knowledge_required"][index] = bool(recipe.knowledge_required)
    arrays["required_memories_level"][index] = memories_level
    arrays["time_seconds"][index] = time_seconds
    return {
        "recipe_id": recipe.recipe_id,
        "inputs": inputs,
        "outputs": outputs,
        "bench_requirements": requirements,
        "knowledge_required": bool(recipe.knowledge_required),
        "required_memories_level": memories_level,
        "time_seconds": float(time_seconds),
    }


def _material_manifest(
    material: ResolvedMaterial,
    *,
    item_id_resolver: SemanticIdResolver,
    resource_type_id_resolver: SemanticIdResolver,
    output: bool,
) -> dict[str, object]:
    if not isinstance(material, ResolvedMaterial):
        raise TypeError("recipe materials must be ResolvedMaterial values")
    quantity = _exact_integer(
        material.quantity,
        "resolved material quantity",
        minimum=1,
        maximum=_INT32_MAX,
    )
    item = _optional_identity(material.item_asset_id, "item_asset_id")
    resource = _optional_identity(material.resource_type_id, "resource_type_id")
    if item is None and resource is None:
        raise ValueError("resolved material needs an item or resource selector")
    if output and (item is None or resource is not None):
        raise ValueError("resolved outputs must identify one concrete item only")
    metadata = _metadata_hash(material.metadata_hash)
    return {
        "item_asset_id": item,
        "item_id": (
            ABSENT_ID
            if item is None
            else _semantic_id(item_id_resolver(item), "item_id_resolver")
        ),
        "resource_type": resource,
        "resource_type_id": (
            ABSENT_ID
            if resource is None
            else _semantic_id(
                resource_type_id_resolver(resource),
                "resource_type_id_resolver",
            )
        ),
        "quantity": quantity,
        "metadata_hash": metadata,
    }


def _requirement_manifest(
    requirement: ResolvedBenchRequirement,
) -> dict[str, object]:
    if not isinstance(requirement, ResolvedBenchRequirement):
        raise TypeError("bench requirements must be ResolvedBenchRequirement values")
    bench_type = _exact_integer(
        requirement.bench_type,
        "resolved bench type",
        minimum=0,
        maximum=BENCH_TYPE_COUNT - 1,
    )
    bench_id = _optional_identity(requirement.bench_id, "bench_id")
    if bench_id is None:
        raise ValueError("resolved bench requirement needs a String ID")
    tier = _exact_integer(
        requirement.required_tier_level,
        "resolved bench tier",
        minimum=0,
        maximum=_INT32_MAX,
    )
    return {
        "bench_type": bench_type,
        "bench_id": bench_id,
        "required_tier_level": tier,
    }


def _empty_table_arrays() -> dict[str, np.ndarray]:
    recipe = (RECIPE_TABLE_CAPACITY,)
    inputs = recipe + (MAX_INGREDIENTS,)
    outputs = recipe + (MAX_OUTPUTS,)
    requirements = recipe + (MAX_BENCH_REQUIREMENTS,)
    return {
        "recipe_mask": np.zeros(recipe, dtype=np.bool_),
        "recipe_id_hash": np.zeros(
            recipe + (IDENTITY_HASH_WORDS,),
            dtype=np.uint32,
        ),
        "input_mask": np.zeros(inputs, dtype=np.bool_),
        "input_item_id": np.full(inputs, ABSENT_ID, dtype=np.int32),
        "input_resource_type_id": np.full(inputs, ABSENT_ID, dtype=np.int32),
        "input_quantity": np.zeros(inputs, dtype=np.int32),
        "input_metadata_required": np.zeros(inputs, dtype=np.bool_),
        "input_metadata_hash": np.zeros(
            inputs + (METADATA_HASH_WORDS,),
            dtype=np.uint32,
        ),
        "output_mask": np.zeros(outputs, dtype=np.bool_),
        "output_item_id": np.full(outputs, ABSENT_ID, dtype=np.int32),
        "output_quantity": np.zeros(outputs, dtype=np.int32),
        "output_metadata_hash": np.zeros(
            outputs + (METADATA_HASH_WORDS,),
            dtype=np.uint32,
        ),
        "requirement_mask": np.zeros(requirements, dtype=np.bool_),
        "requirement_bench_type": np.full(
            requirements,
            ABSENT_ID,
            dtype=np.int32,
        ),
        "requirement_bench_id_hash": np.zeros(
            requirements + (IDENTITY_HASH_WORDS,),
            dtype=np.uint32,
        ),
        "requirement_tier_level": np.zeros(requirements, dtype=np.int32),
        "knowledge_required": np.zeros(recipe, dtype=np.bool_),
        "required_memories_level": np.ones(recipe, dtype=np.int32),
        "time_seconds": np.zeros(recipe, dtype=np.float32),
    }


def _semantic_id(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must return an integer")
    resolved = int(value)
    if not 0 <= resolved <= _INT32_MAX:
        raise ValueError(f"{name} returned a value outside nonnegative int32")
    return resolved


def _metadata_hash(value: tuple[int, ...]) -> tuple[int, ...]:
    if len(value) != METADATA_HASH_WORDS:
        raise ValueError(f"metadata_hash must have {METADATA_HASH_WORDS} uint32 words")
    return tuple(
        _exact_integer(
            word,
            "metadata_hash word",
            minimum=0,
            maximum=np.iinfo(np.uint32).max,
        )
        for word in value
    )


def _optional_identity(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be None or a non-empty string")
    return value


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must contain hexadecimal characters") from error
    return value.upper()


def _exact_integer(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    resolved = int(value)
    if not minimum <= resolved <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return resolved


def _float32(
    value: object,
    name: str,
    *,
    minimum: float,
) -> np.float32:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    packed = np.float32(resolved)
    if not np.isfinite(packed):
        raise ValueError(f"{name} must fit float32")
    return packed


def _validate_semantic_identity_mapping(
    recipes: list[dict[str, object]],
) -> None:
    item_to_id: dict[str, int] = {}
    id_to_item: dict[int, str] = {}
    resource_to_id: dict[str, int] = {}
    id_to_resource: dict[int, str] = {}

    def register(
        identity: object,
        semantic_id: object,
        *,
        identity_to_id: dict[str, int],
        id_to_identity: dict[int, str],
        domain: str,
    ) -> None:
        if identity is None:
            return
        native = str(identity)
        resolved = int(semantic_id)
        if native in identity_to_id and identity_to_id[native] != resolved:
            raise ValueError(f"{domain} resolver is not deterministic for {native!r}")
        if resolved in id_to_identity and id_to_identity[resolved] != native:
            raise ValueError(
                f"{domain} semantic ID collision between "
                f"{id_to_identity[resolved]!r} and {native!r}"
            )
        identity_to_id[native] = resolved
        id_to_identity[resolved] = native

    for recipe in recipes:
        materials = tuple(recipe["inputs"]) + tuple(recipe["outputs"])
        for material in materials:
            register(
                material["item_asset_id"],
                material["item_id"],
                identity_to_id=item_to_id,
                id_to_identity=id_to_item,
                domain="item",
            )
            register(
                material["resource_type"],
                material["resource_type_id"],
                identity_to_id=resource_to_id,
                id_to_identity=id_to_resource,
                domain="resource",
            )


__all__ = [
    "PackedRecipeTable",
    "ResolvedBenchRequirement",
    "ResolvedMaterial",
    "ResolvedRecipe",
    "SemanticIdResolver",
    "pack_resolved_recipe_table",
]
