"""Compile and execute lossless fieldcraft actions in Region environments."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.inventory import (
    CONTAINER_BACKPACK,
    CONTAINER_HOTBAR,
    CONTAINER_STORAGE,
    InventoryLayout,
    InventoryState,
)
from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.crafting import (
    ABSENT_ID,
    BENCH_TYPE_CRAFTING,
    INPUT_MODE_NORMAL,
    MAX_OUTPUTS,
    MAX_RESOURCE_TYPES_PER_ITEM,
    RECIPE_TABLE_CAPACITY,
    CombatCraftingCatalog,
    CraftingContext,
    CraftingInventory,
    PackedRecipeTable,
    ResolvedBenchRequirement,
    ResolvedMaterial,
    ResolvedRecipe,
    craft_into_combat_inventory,
    identity_sha256_words,
    pack_resolved_recipe_table,
    player_crafting_input_slot_order,
    satisfiable_recipes,
)
from hytalegym.jax.world import (
    ActorRecipeCandidates,
    RegionActionRuntimeState,
    produce_actor_recipe_candidates_from_legal_mask,
    requery_region_block_action_candidates,
)
from hytalegym.worldgen import block_semantic_key
from hytalegym.worldgen.native_crafting import NativeCraftingRecipe
from hytalegym.worldgen.surrogate import (
    HytaleAssetArchive,
    LocalBlockSemanticsResolver,
    default_hytale_assets_path,
)


REGION_FIELDCRAFT_SCHEMA = "hytalerl_region_fieldcraft_v3"
REGION_FIELDCRAFT_VERSION = 3
_FIELDCRAFT_ID = "Fieldcraft"
_BENCH_TYPE_NAMES = {
    "Crafting": BENCH_TYPE_CRAFTING,
}
_BLOCK_ROTATION_COUNT = 64


class RegionFieldcraftAssets(NamedTuple):
    """Static lossless recipes, ordinary benches, and item properties."""

    packed: PackedRecipeTable
    output_max_stack: jax.Array
    output_max_durability: jax.Array
    inventory_layout: InventoryLayout
    source_bridge_sha256: str
    bench_semantic_key: jax.Array
    bench_id_hash: jax.Array
    bench_type: jax.Array
    bench_tier_level: jax.Array
    bench_valid: jax.Array


class RegionCraftExecution(NamedTuple):
    state: InventoryState
    requested: jax.Array
    target_rechecked: jax.Array
    accepted: jax.Array
    finished: jax.Array
    recipe_index: jax.Array


def region_fieldcraft_contract_sha256() -> str:
    payload = {
        "schema": REGION_FIELDCRAFT_SCHEMA,
        "version": REGION_FIELDCRAFT_VERSION,
        "source": "native_runtime_resolved_crafting_catalog_v2",
        "admission": {
            "bench": (
                "native_requirement_set_includes_Crafting_Fieldcraft_or_"
                "an_exact_visible_requeried_installed_Crafting_bench_at_"
                "tier_zero"
            ),
            "materials": "concrete_item_only_no_metadata_tags_or_exclusions",
            "knowledge": "not_required_memories_level_one",
            "duration_seconds": 0,
        },
        "actor_rows": (
            "controlled_actor_zero_inventory_and_candidate_row_only_"
            "extra_entities_excluded"
        ),
        "candidate_policy": "complete_legal_set_or_fail_closed_at_16",
        "execution": "atomic_zero_duration_inventory_transaction",
        "overflow": "fail_closed_no_unpublished_world_drop",
        "ordinary_bench_identity": (
            "installed_BlockType_Bench_Type_Crafting_String_Id_and_rotation"
        ),
        "specialized_benches": (
            "processing_diagram_structural_upgraded_and_timed_future_not_claimed"
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest().upper()


def compile_region_fieldcraft_assets(
    recording_path: str | Path,
    inventory_layout: InventoryLayout,
    *,
    expected_bridge_sha256: str,
    assets_path: str | Path | None = None,
) -> RegionFieldcraftAssets:
    """Compile the complete losslessly representable zero-time fieldcraft set."""

    if not isinstance(inventory_layout, InventoryLayout):
        raise TypeError("inventory_layout has the wrong type")
    value = json.loads(Path(recording_path).read_text(encoding="utf-8"))
    bridge = _sha256(value.get("bridge_sha256"), "bridge_sha256")
    if bridge != _sha256(expected_bridge_sha256, "expected_bridge_sha256"):
        raise ValueError("crafting recording came from a different bridge")
    rows = tuple(
        NativeCraftingRecipe.from_response(row)
        for row in value.get("recipes", ())
    )
    archive_path = (
        default_hytale_assets_path() if assets_path is None else Path(assets_path)
    )
    with HytaleAssetArchive(archive_path) as archive:
        resolver = LocalBlockSemanticsResolver(archive)
        benches = _ordinary_crafting_benches(archive, resolver)
        admitted_bench_ids = {_FIELDCRAFT_ID, *(row[1] for row in benches)}
        selected = tuple(
            row
            for row in rows
            if _lossless_region_crafting(row, admitted_bench_ids)
        )
        properties = _output_item_properties(selected, resolver)
    if not selected:
        raise ValueError(
            "native catalog has no lossless zero-time Region crafting rows"
        )
    resolved = tuple(_resolved_recipe(row) for row in selected)
    packed = pack_resolved_recipe_table(
        resolved,
        source_sha256=_sha256(
            value.get("catalog_semantic_sha256"),
            "catalog_semantic_sha256",
        ),
        expected_recipe_count=len(resolved),
        item_id_resolver=semantic_id,
        resource_type_id_resolver=semantic_id,
    )
    maximum_stack = np.zeros((RECIPE_TABLE_CAPACITY, MAX_OUTPUTS), dtype=np.int32)
    maximum_durability = np.zeros(
        (RECIPE_TABLE_CAPACITY, MAX_OUTPUTS),
        dtype=np.float32,
    )
    for row_index, row in enumerate(selected):
        for output_index, material in enumerate(row.outputs):
            item_id = material.item_asset_id
            maximum_stack[row_index, output_index], maximum_durability[
                row_index, output_index
            ] = properties[item_id]
    bench_rows = [
        (asset_id, bench_id, bench_type, rotation)
        for asset_id, bench_id, bench_type in benches
        for rotation in range(_BLOCK_ROTATION_COUNT)
    ]
    return RegionFieldcraftAssets(
        packed=packed,
        output_max_stack=jnp.asarray(maximum_stack),
        output_max_durability=jnp.asarray(maximum_durability),
        inventory_layout=inventory_layout,
        source_bridge_sha256=bridge,
        bench_semantic_key=jnp.asarray(
            [block_semantic_key(asset_id, rotation)
             for asset_id, _bench_id, _bench_type, rotation in bench_rows],
            dtype=jnp.uint32,
        ),
        bench_id_hash=jnp.asarray(
            [identity_sha256_words(bench_id)
             for _asset_id, bench_id, _bench_type, _rotation in bench_rows],
            dtype=jnp.uint32,
        ),
        bench_type=jnp.asarray(
            [bench_type
             for _asset_id, _bench_id, bench_type, _rotation in bench_rows],
            dtype=jnp.int32,
        ),
        bench_tier_level=jnp.zeros((len(bench_rows),), dtype=jnp.int32),
        bench_valid=jnp.ones((len(bench_rows),), dtype=jnp.bool_),
    )


def make_region_fieldcraft_candidate_provider(
    assets: RegionFieldcraftAssets,
    *,
    block_candidate_provider=None,
):
    """Return JIT-safe Fieldcraft plus exact ordinary-bench candidates."""

    if not isinstance(assets, RegionFieldcraftAssets):
        raise TypeError("assets has the wrong type")

    def provider(state, _world, _params):
        inventory = _crafting_inventory(
            _actor_inventory(state.inventory),
            assets.inventory_layout,
        )
        batch, entities = inventory.item_id.shape[:2]
        fieldcraft = _fieldcraft_context(batch, entities)
        legal = _legal_recipes(inventory, fieldcraft, assets.packed.table)
        evidence = jnp.ones((batch, entities), dtype=jnp.bool_)
        if block_candidate_provider is not None:
            if not isinstance(_world, RegionActionRuntimeState):
                raise TypeError(
                    "ordinary bench crafting requires RegionActionRuntimeState"
                )
            blocks, _maximum_distance = block_candidate_provider(
                state,
                _world,
                _params,
            )
            blocks = _controlled_block_candidates(blocks, entities)
            current = requery_region_block_action_candidates(
                _world.geometry,
                _world.semantic_atlas,
                blocks,
            )
            candidate_count = blocks.candidate_mask.shape[2]
            current = jax.tree.map(
                lambda value: value.reshape(
                    (batch, entities, candidate_count) + value.shape[2:]
                ),
                current,
            )
            bench = _bench_contexts(assets, blocks, current)
            legal = legal | jnp.any(
                _legal_recipes(
                    _repeat_inventory_for_candidates(inventory, candidate_count),
                    bench,
                    assets.packed.table,
                )
                & bench.window_open[..., None],
                axis=2,
            )
        return produce_actor_recipe_candidates_from_legal_mask(
            assets.packed.table,
            legal,
            catalog_available=True,
            legal_evidence_available=evidence,
        )

    provider.contract_sha256 = region_fieldcraft_contract_sha256()
    provider.catalog_content_sha256 = assets.packed.content_sha256
    return provider


def execute_region_fieldcraft(
    assets: RegionFieldcraftAssets,
    state: InventoryState,
    candidates: ActorRecipeCandidates,
    candidate_index: jax.Array,
) -> RegionCraftExecution:
    """Recheck one actor candidate and atomically commit a zero-time craft."""

    indexes = jnp.asarray(candidate_index, dtype=jnp.int32)
    if indexes.ndim != 1:
        raise ValueError("candidate_index must have shape [batch]")
    batch = indexes.shape[0]
    safe = jnp.clip(indexes, 0, candidates.candidate_mask.shape[2] - 1)
    actor_available = candidates.available[:, 0]
    candidate_mask = jnp.take_along_axis(
        candidates.candidate_mask[:, 0],
        safe[:, None],
        axis=1,
    )[:, 0]
    recipe_index = jnp.take_along_axis(
        candidates.execution.recipe_index[:, 0],
        safe[:, None],
        axis=1,
    )[:, 0]
    requested = indexes >= 0
    rechecked = requested & actor_available & candidate_mask & (recipe_index >= 0)
    actor = _actor_inventory(state)
    output_stack = assets.output_max_stack[recipe_index][:, None, :]
    output_durability = assets.output_max_durability[recipe_index][:, None, :]
    slot_count = state.item_id.shape[2]
    allowed = jnp.broadcast_to(
        jnp.isin(
            assets.inventory_layout.container_id,
            jnp.asarray(
                (CONTAINER_HOTBAR, CONTAINER_STORAGE, CONTAINER_BACKPACK),
                dtype=jnp.int32,
            ),
        )[None, None, None, :],
        (batch, 1, MAX_OUTPUTS, slot_count),
    )
    catalog = CombatCraftingCatalog(
        slot_resource_type_id=jnp.full(
            (batch, 1, slot_count, MAX_RESOURCE_TYPES_PER_ITEM),
            ABSENT_ID,
            dtype=jnp.int32,
        ),
        output_max_stack=output_stack,
        output_max_durability=output_durability,
        output_container_priority=jnp.broadcast_to(
            jnp.asarray(
                (CONTAINER_HOTBAR, CONTAINER_STORAGE, CONTAINER_BACKPACK),
                dtype=jnp.int32,
            )[None, None, None, :],
            (batch, 1, MAX_OUTPUTS, 3),
        ),
        output_slot_allowed=allowed,
    )
    transition = craft_into_combat_inventory(
        assets.packed.table,
        actor,
        assets.inventory_layout,
        _fieldcraft_context(batch, 1),
        recipe_index[:, None],
        catalog,
        quantity=rechecked[:, None].astype(jnp.int32),
        input_slot_order=player_crafting_input_slot_order(assets.inventory_layout),
    )
    no_drop = ~jnp.any(transition.world_drop_mask[:, 0], axis=1)
    accepted = rechecked & transition.crafting.crafted[:, 0] & no_drop
    committed = _merge_actor_inventory(state, transition.state)
    committed = jax.tree.map(
        lambda next_value, old_value: jnp.where(
            accepted.reshape((batch,) + (1,) * (next_value.ndim - 1)),
            next_value,
            old_value,
        ),
        committed,
        state,
    )
    return RegionCraftExecution(
        state=committed,
        requested=requested,
        target_rechecked=(~requested) | rechecked,
        accepted=accepted,
        finished=accepted,
        recipe_index=jnp.where(rechecked, recipe_index, jnp.int32(-1)),
    )


def _lossless_region_crafting(
    recipe: NativeCraftingRecipe,
    admitted_bench_ids: set[str],
) -> bool:
    return bool(
        not recipe.knowledge_required
        and recipe.required_memories_level == 1
        and recipe.time_seconds == 0.0
        and any(
            requirement.bench_type == BENCH_TYPE_CRAFTING
            and requirement.bench_id in admitted_bench_ids
            and requirement.required_tier_level == 0
            for requirement in recipe.bench_requirements
        )
        and all(_concrete(material) for material in (*recipe.inputs, *recipe.outputs))
    )


def _ordinary_crafting_benches(
    archive: HytaleAssetArchive,
    resolver: LocalBlockSemanticsResolver,
) -> tuple[tuple[str, str, int], ...]:
    """Resolve installed ordinary Crafting benches without name branches."""

    rows: list[tuple[str, str, int]] = []
    for asset_id in archive.item_asset_ids():
        item, _provenance = resolver.load_inherited_item_asset(asset_id)
        block = item.get("BlockType")
        bench = block.get("Bench") if isinstance(block, dict) else None
        if not isinstance(bench, dict):
            continue
        type_name = bench.get("Type")
        if type_name not in _BENCH_TYPE_NAMES:
            continue
        bench_id = bench.get("Id")
        if not isinstance(bench_id, str) or not bench_id:
            raise ValueError(f"bench {asset_id!r} has no String ID")
        rows.append((asset_id, bench_id, _BENCH_TYPE_NAMES[type_name]))
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("installed ordinary bench asset IDs are not unique")
    return tuple(sorted(rows))


def _output_item_properties(
    recipes: tuple[NativeCraftingRecipe, ...],
    resolver: LocalBlockSemanticsResolver,
) -> dict[str, tuple[int, float]]:
    properties: dict[str, tuple[int, float]] = {}
    for row in recipes:
        for material in row.outputs:
            item_id = material.item_asset_id
            if item_id not in properties:
                item, _provenance = resolver.load_inherited_item_asset(item_id)
                properties[item_id] = _item_properties(item)
    return properties


def _concrete(material) -> bool:
    return bool(
        material.item_asset_id
        and not material.resource_type_id
        and not material.item_tag_present
        and not material.excluded_item_asset_ids
        and not material.metadata_json_sha256
    )


def _resolved_recipe(recipe: NativeCraftingRecipe) -> ResolvedRecipe:
    return ResolvedRecipe(
        recipe_id=recipe.recipe_id,
        inputs=tuple(
            ResolvedMaterial(value.quantity, item_asset_id=value.item_asset_id)
            for value in recipe.inputs
        ),
        outputs=tuple(
            ResolvedMaterial(value.quantity, item_asset_id=value.item_asset_id)
            for value in recipe.outputs
        ),
        bench_requirements=tuple(
            ResolvedBenchRequirement(
                value.bench_type,
                value.bench_id,
                value.required_tier_level,
            )
            for value in recipe.bench_requirements
        ),
        knowledge_required=False,
        required_memories_level=1,
        time_seconds=0.0,
    )


def _crafting_inventory(
    state: InventoryState,
    layout: InventoryLayout,
) -> CraftingInventory:
    """Expose exactly the containers native player crafting consumes.

    ``NativeHeadlessCrafting`` uses ``BACKPACK_STORAGE_HOTBAR``. Counting
    armor, utility, or tool slots here advertises a recipe that the native
    commit path can reject with unchanged inventory. Keep the policy legality
    predicate and commit-time container boundary identical.
    """

    if not isinstance(layout, InventoryLayout):
        raise TypeError("layout has the wrong type")
    slot_count = state.item_id.shape[-1]
    if layout.container_id.shape != (slot_count,):
        raise ValueError("inventory state and layout slot axes differ")
    input_slot = jnp.isin(
        layout.container_id,
        jnp.asarray(
            (CONTAINER_BACKPACK, CONTAINER_STORAGE, CONTAINER_HOTBAR),
            dtype=jnp.int32,
        ),
    )
    present = input_slot.reshape((1,) * (state.item_id.ndim - 1) + (slot_count,))
    shape = state.item_id.shape + (MAX_RESOURCE_TYPES_PER_ITEM,)
    return CraftingInventory(
        item_id=jnp.where(present, state.item_id, ABSENT_ID),
        quantity=jnp.where(present, state.quantity, 0),
        resource_type_id=jnp.full(shape, ABSENT_ID, dtype=jnp.int32),
        metadata_hash=jnp.where(present[..., None], state.metadata_hash, 0),
    )


def _controlled_block_candidates(candidates, actor_count: int):
    """Keep only candidate rows owned by actor-visible inventory state."""

    if (
        isinstance(actor_count, bool)
        or not isinstance(actor_count, int)
        or actor_count < 1
    ):
        raise ValueError("actor_count must be a positive integer")
    batch, candidate_actors = candidates.candidate_mask.shape[:2]
    if actor_count > candidate_actors:
        raise ValueError(
            "block candidates contain fewer actor rows than inventory state"
        )

    def controlled(value):
        if value.ndim < 2 or value.shape[:2] != (batch, candidate_actors):
            raise ValueError(
                "every block-candidate field must share [batch, actor] axes"
            )
        return value[:, :actor_count, ...]

    return jax.tree.map(controlled, candidates)


def _fieldcraft_context(batch: int, entities: int) -> CraftingContext:
    prefix = (batch, entities)
    return CraftingContext(
        window_open=jnp.ones(prefix, dtype=jnp.bool_),
        bench_type=jnp.full(prefix, BENCH_TYPE_CRAFTING, dtype=jnp.int32),
        bench_id_hash=jnp.broadcast_to(
            jnp.asarray(identity_sha256_words(_FIELDCRAFT_ID), dtype=jnp.uint32),
            prefix + (8,),
        ),
        bench_tier_level=jnp.zeros(prefix, dtype=jnp.int32),
        memories_level=jnp.ones(prefix, dtype=jnp.int32),
        known_recipe=jnp.ones(prefix + (RECIPE_TABLE_CAPACITY,), dtype=jnp.bool_),
        creative_mode=jnp.zeros(prefix, dtype=jnp.bool_),
        input_mode=jnp.full(prefix, INPUT_MODE_NORMAL, dtype=jnp.int32),
    )


def _legal_recipes(
    inventory: CraftingInventory,
    context: CraftingContext,
    table,
) -> jax.Array:
    prefix = context.window_open.shape
    flat_inventory = jax.tree.map(
        lambda value: value.reshape((-1,) + value.shape[len(prefix):]),
        inventory,
    )
    flat_context = jax.tree.map(
        lambda value: value.reshape((-1,) + value.shape[len(prefix):]),
        context,
    )
    legal = jax.vmap(
        lambda one_inventory, one_context: satisfiable_recipes(
            table,
            one_inventory,
            one_context,
        )
    )(flat_inventory, flat_context)
    return legal.reshape(prefix + (RECIPE_TABLE_CAPACITY,))


def _repeat_inventory_for_candidates(
    inventory: CraftingInventory,
    candidate_count: int,
) -> CraftingInventory:
    return jax.tree.map(
        lambda value: jnp.broadcast_to(
            value[:, :, None, ...],
            value.shape[:2] + (candidate_count,) + value.shape[2:],
        ),
        inventory,
    )


def _bench_contexts(
    assets: RegionFieldcraftAssets,
    candidates,
    current,
) -> CraftingContext:
    keys = current.geometry.semantic_key
    matches = (
        assets.bench_valid[None, None, None, :]
        & current.geometry.semantic_key_valid[..., None]
        & jnp.all(
            keys[..., None, :] == assets.bench_semantic_key[None, None, None],
            axis=-1,
        )
    )
    unique = jnp.sum(matches.astype(jnp.int32), axis=-1) == 1
    row = jnp.argmax(matches, axis=-1).astype(jnp.int32)
    available = (
        candidates.available[..., None]
        & candidates.candidate_mask
        & current.available
        & current.geometry.exact
        & ~current.invalid
        & ~current.resync_required
        & unique
    )
    prefix = available.shape
    return CraftingContext(
        window_open=available,
        bench_type=jnp.where(available, assets.bench_type[row], 0),
        bench_id_hash=jnp.where(
            available[..., None],
            assets.bench_id_hash[row],
            jnp.zeros(prefix + (8,), dtype=jnp.uint32),
        ),
        bench_tier_level=jnp.where(
            available,
            assets.bench_tier_level[row],
            0,
        ),
        memories_level=jnp.ones(prefix, dtype=jnp.int32),
        known_recipe=jnp.ones(
            prefix + (RECIPE_TABLE_CAPACITY,),
            dtype=jnp.bool_,
        ),
        creative_mode=jnp.zeros(prefix, dtype=jnp.bool_),
        input_mode=jnp.full(prefix, INPUT_MODE_NORMAL, dtype=jnp.int32),
    )


def _actor_inventory(state: InventoryState) -> InventoryState:
    return InventoryState(**{
        name: value if name == "failure_bits" else value[:, :1, ...]
        for name, value in state._asdict().items()
    })


def _merge_actor_inventory(full: InventoryState, actor: InventoryState) -> InventoryState:
    return InventoryState(**{
        name: (
            actor.failure_bits
            if name == "failure_bits"
            else value.at[:, 0, ...].set(getattr(actor, name)[:, 0, ...])
        )
        for name, value in full._asdict().items()
    })


def _item_properties(item) -> tuple[int, float]:
    authored = item.get("MaxStack")
    if authored is None:
        unique = any(
            item.get(name) is not None
            for name in ("Tool", "Weapon", "Armor", "BuilderTool", "BlockSelectorTool")
        )
        maximum_stack = 1 if unique else 100
    elif isinstance(authored, bool) or not isinstance(authored, int) or authored < 1:
        raise ValueError("Item.MaxStack must be a positive integer")
    else:
        maximum_stack = authored
    durability = item.get("MaxDurability", 0.0)
    if isinstance(durability, bool) or not isinstance(durability, (int, float)):
        raise ValueError("Item.MaxDurability must be numeric")
    maximum_durability = float(durability)
    if not math.isfinite(maximum_durability) or maximum_durability < 0.0:
        raise ValueError("Item.MaxDurability must be finite and nonnegative")
    return maximum_stack, maximum_durability


def _sha256(value, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value.upper()


__all__ = [
    "REGION_FIELDCRAFT_SCHEMA",
    "REGION_FIELDCRAFT_VERSION",
    "RegionCraftExecution",
    "RegionFieldcraftAssets",
    "compile_region_fieldcraft_assets",
    "execute_region_fieldcraft",
    "make_region_fieldcraft_candidate_provider",
    "region_fieldcraft_contract_sha256",
]
