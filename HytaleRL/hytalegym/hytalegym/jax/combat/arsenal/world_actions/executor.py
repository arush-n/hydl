"""Compose exact Region-backed execution for Arsenal world actions."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.arsenal.environment import (
    ACTION_SURFACE_REJECT_EXECUTOR_DENIED,
    ACTION_SURFACE_VERB_BLOCK_INTERACTION,
    ACTION_SURFACE_VERB_CRAFT,
    ACTION_SURFACE_VERB_COUNT,
    ACTION_SURFACE_VERB_USE,
    ArsenalActionSurfaceExecution,
    ArsenalBlockCandidateProvider,
    action_surface_lifecycle_evidence,
    action_surface_verb_requests,
)
from hytalegym.jax.combat.arsenal.world_actions.crafting import (
    RegionFieldcraftAssets,
    execute_region_fieldcraft,
    region_fieldcraft_contract_sha256,
)
from hytalegym.jax.combat.arsenal.runtime import inventory_from_loadout
from hytalegym.jax.combat.block_interactions import (
    BLOCK_INTERACTION_BREAK,
    BLOCK_INTERACTION_PLACE,
    BLOCK_INTERACTION_RUNNING,
    BLOCK_INTERACTION_SUCCEEDED,
    NO_INTERACTION_PROGRAM,
    BlockInteractionCommand,
    BlockInteractionProgram,
    BlockInteractionState,
    block_interaction_program_from_native_node,
    empty_block_interaction_program,
    step_block_interactions_from_world,
    block_interaction_contract_sha256,
)
from hytalegym.jax.combat.inventory import (
    CONTAINER_HOTBAR,
    InventoryLayout,
    InventoryState,
    item_stack_at,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_BLOCK_TRIGGER_PRIMARY,
    ARSENAL_BLOCK_TRIGGER_SECONDARY,
)
from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.types import AGENT_ENTITY
from hytalegym.jax.world import (
    BlockActionCatalog,
    BlockDropProgramCatalog,
    BlockGatherDefaults,
    BlockToolState,
    RegionActionRuntimeState,
    RegionLibraryEnvironmentSelection,
    RegionStateChangeUseTable,
    block_action_catalog_from_local,
    block_drop_program_catalog_from_local,
    block_gather_defaults_from_local,
    block_tool_state_from_local,
    empty_mutable_block_geometry,
    compile_region_state_change_use_table,
    execute_region_state_change_use,
    execute_region_runtime_block_action_with_placement,
    produce_region_runtime_actor_block_placement_candidates,
    region_block_placement_support_result,
    region_removal_cascade_safety,
    region_state_change_use_candidates,
    region_state_change_use_runtime_contract_sha256,
    select_region_block_action_target,
)
from hytalegym.worldgen import (
    INTERACTION_TYPE_NAMES,
    NATIVE_ITEM_TRIGGER_CAPACITY,
    NativeItemInteractionEvidence,
    NativePlaceBlockPayload,
    block_action_contract_sha256,
    block_drop_program_contract_sha256,
    block_semantic_key,
    resolve_local_block_affordance,
    resolve_local_held_item_tool,
    stable_asset_id,
    native_item_interaction_contract_sha256,
)
from hytalegym.worldgen.region import (
    BLOCK_SEMANTIC_KEY_WORDS,
    RegionArtifactLibrary,
    RegionBlockSemanticLibrary,
    compile_region_state_change_use_definitions,
)
from hytalegym.worldgen.region.block_action_catalog import (
    compile_region_block_action_catalog,
    resolve_region_block_asset_references,
)
from hytalegym.worldgen.surrogate import (
    HytaleAssetArchive,
    LocalBlockSemanticsResolver,
    default_hytale_assets_path,
)
from hytalegym.jax.combat.arsenal.world_actions.placement import (
    RegionPlacedGeometryTable,
    compile_region_placed_geometry_table,
    region_placed_geometry_contract_sha256,
    select_actor_aim_block_face,
    select_region_placed_geometry,
)


REGION_BLOCK_EXECUTOR_SCHEMA = "hytalerl_region_block_executor_v4"
REGION_BLOCK_EXECUTOR_VERSION = 4


def region_block_executor_contract_sha256() -> str:
    """Return the semantic identity of the fail-closed executor boundary."""

    payload = {
        "schema": REGION_BLOCK_EXECUTOR_SCHEMA,
        "version": REGION_BLOCK_EXECUTOR_VERSION,
        "native_item_interaction_contract_sha256": (
            native_item_interaction_contract_sha256()
        ),
        "block_interaction_contract_sha256": (
            block_interaction_contract_sha256()
        ),
        "block_action_contract_sha256": block_action_contract_sha256(),
        "drop_program_contract_sha256": block_drop_program_contract_sha256(),
        "placed_geometry_contract_sha256": (
            region_placed_geometry_contract_sha256()
        ),
        "region_fieldcraft_contract_sha256": (
            region_fieldcraft_contract_sha256()
        ),
        "region_state_change_use_runtime_contract_sha256": (
            region_state_change_use_runtime_contract_sha256()
        ),
        "supported_leaf": (
            "unique_primary_or_secondary_break_or_face_correct_place"
        ),
        "outer_root_graph": (
            "not_executed_block_head_selects_the_validated_leaf_directly"
        ),
        "target_recheck": "current_region_runtime_revision",
        "mutation_commit": "acknowledged_sparse_overlay_then_world_reprojection",
        "supported_place": {
            "face": "current_actor_aim_unique_unit_cell_entry_face",
            "destination": "exact_empty_current_region_requery",
            "geometry": "exact_matching_semantic_row_from_loaded_region",
            "support": (
                "native_BlockPlacementHelper_testSupportingBlock_certificate"
            ),
        },
        "supported_craft": (
            "optional_complete_lossless_zero_time_fieldcraft_candidate_set"
        ),
        "supported_use": (
            "unarmed_actor_legal_exact_one_node_ChangeState_with_same_"
            "region_exact_target_geometry"
        ),
        "unsupported_use": (
            "doors_multi_node_UpdateBlockState_true_missing_target_geometry_"
            "or_stale_current_state"
        ),
        "stale_region_asset_semantics": "explicitly_excluded",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest().upper()


class RegionBlockInteractionTable(NamedTuple):
    """Fixed item/trigger bank compiled from validated native graphs."""

    item_id: jax.Array
    valid: jax.Array
    trigger_valid: jax.Array
    program: BlockInteractionProgram
    break_tool_type_id: jax.Array
    break_match_tool: jax.Array
    place_semantic_key: jax.Array
    place_semantic_key_valid: jax.Array


class RegionHeldToolTable(NamedTuple):
    """Source-resolved held-tool rows keyed by Combat semantic item ID."""

    item_id: jax.Array
    valid: jax.Array
    tool: BlockToolState


class RegionBlockExecutorAssets(NamedTuple):
    """Host-compiled action inputs and explicit stale-entry exclusions."""

    interactions: RegionBlockInteractionTable
    held_tools: RegionHeldToolTable
    block_catalog: BlockActionCatalog
    gather_defaults: BlockGatherDefaults
    drop_programs: BlockDropProgramCatalog | None
    placed_geometry: RegionPlacedGeometryTable
    state_change_use: RegionStateChangeUseTable | None
    excluded_asset_references: tuple[str, ...]


def make_region_block_item_availability_provider(
    interaction_table: RegionBlockInteractionTable,
    inventory_layout: InventoryLayout,
    *,
    block_candidate_provider: ArsenalBlockCandidateProvider | None = None,
    state_change_use_table: RegionStateChangeUseTable | None = None,
):
    """Expose only item roots this executor can commit end to end."""

    if not isinstance(interaction_table, RegionBlockInteractionTable):
        raise TypeError("interaction_table has the wrong type")
    if not isinstance(inventory_layout, InventoryLayout):
        raise TypeError("inventory_layout has the wrong type")
    if state_change_use_table is not None and block_candidate_provider is None:
        raise ValueError(
            "state_change_use_table requires block_candidate_provider"
        )
    if block_candidate_provider is not None and not callable(
        block_candidate_provider
    ):
        raise TypeError("block_candidate_provider must be callable")
    primary = INTERACTION_TYPE_NAMES.index("Primary")
    secondary = INTERACTION_TYPE_NAMES.index("Secondary")
    use = INTERACTION_TYPE_NAMES.index("Use")

    def provider(state, world, params):
        stack = item_stack_at(
            state.inventory,
            inventory_layout,
            CONTAINER_HOTBAR,
            state.inventory.active_hotbar_slot,
        )
        actor_item = stack.item_id[:, 0]
        row, unique = _unique_row(
            interaction_table.item_id,
            interaction_table.valid,
            actor_item,
        )
        kind = interaction_table.program.kind[row]
        executable = (
            interaction_table.trigger_valid[row]
            & (
                (kind == BLOCK_INTERACTION_BREAK)
                | (
                    (kind == BLOCK_INTERACTION_PLACE)
                    & interaction_table.place_semantic_key_valid[row]
                )
            )
            & unique[:, None]
            & (stack.quantity[:, 0, None] > 0)
        )
        result = jnp.zeros(
            (actor_item.shape[0], NATIVE_ITEM_TRIGGER_CAPACITY),
            dtype=jnp.bool_,
        ).at[:, primary].set(executable[:, 0]).at[:, secondary].set(
            executable[:, 1]
        )
        if state_change_use_table is not None:
            candidates, _ = block_candidate_provider(state, world, params)
            use_candidates = region_state_change_use_candidates(
                world,
                candidates,
                state_change_use_table,
            )
            result = result.at[:, use].set(
                use_candidates.available[:, 0]
                & (stack.quantity[:, 0] <= 0)
            )
        return result

    return provider


def make_region_held_item_inventory_reset_provider(
    interaction_table: RegionBlockInteractionTable,
    inventory_layout: InventoryLayout,
    item_asset_id: str,
):
    """Equip one evidence-backed item for an explicit World-action scenario.

    The rest of every entity's authored loadout inventory is preserved. This
    is a scenario input, not an item-specific mechanics branch: the semantic
    item ID must resolve to exactly one validated interaction-evidence row.
    """

    if not isinstance(interaction_table, RegionBlockInteractionTable):
        raise TypeError("interaction_table has the wrong type")
    if not isinstance(inventory_layout, InventoryLayout):
        raise TypeError("inventory_layout has the wrong type")
    if not isinstance(item_asset_id, str) or not item_asset_id:
        raise ValueError("item_asset_id must be a non-empty string")
    item_id = semantic_id(item_asset_id)
    rows = np.asarray(interaction_table.item_id)
    valid = np.asarray(interaction_table.valid)
    if int(np.sum(valid & (rows == item_id))) != 1:
        raise ValueError(
            "scenario held item must identify exactly one compiled "
            "interaction-evidence row"
        )
    hotbar_zero = np.flatnonzero(
        (np.asarray(inventory_layout.container_id) == CONTAINER_HOTBAR)
        & (np.asarray(inventory_layout.container_slot) == 0)
    )
    if hotbar_zero.size != 1:
        raise ValueError("inventory layout must contain one hotbar slot zero")
    inventory_index = int(hotbar_zero[0])

    def provider(_keys, config):
        inventory = inventory_from_loadout(
            config.loadout,
            config.inventory_layout,
        )
        return inventory._replace(
            item_id=inventory.item_id.at[
                :, AGENT_ENTITY, inventory_index
            ].set(jnp.int32(item_id)),
            quantity=inventory.quantity.at[
                :, AGENT_ENTITY, inventory_index
            ].set(jnp.int32(1)),
            active_hotbar_slot=inventory.active_hotbar_slot.at[
                :, AGENT_ENTITY
            ].set(jnp.int32(0)),
        )

    provider.item_asset_id = item_asset_id
    provider.item_semantic_id = item_id
    return provider


def compile_region_block_executor_assets(
    selection: RegionLibraryEnvironmentSelection,
    runtime: RegionActionRuntimeState,
    evidence: Sequence[NativeItemInteractionEvidence],
    *,
    region_manifest_path: str | Path,
    block_semantic_manifest_path: str | Path,
    assets_path: str | Path | None = None,
) -> RegionBlockExecutorAssets:
    """Compile exact action data for a selected frozen Region working set.

    Region sidecars can outlive an installed content update. Entries whose
    captured affordance or semantic key disagrees with the installed asset are
    explicitly excluded, so those cells fail closed instead of inheriting a
    different block's action rules.
    """

    if not isinstance(selection, RegionLibraryEnvironmentSelection):
        raise TypeError("selection must be a RegionLibraryEnvironmentSelection")
    rows = tuple(evidence)
    interactions = region_block_interaction_table_from_native_evidence(rows)
    library = RegionArtifactLibrary.load(region_manifest_path)
    working = selection.working_set
    if library.semantic_sha256 != working.library_semantic_sha256:
        raise ValueError("Region selection names a different artifact library")
    entries = library.select(
        working.split,
        len(working.artifact_semantic_sha256),
        selection_key=working.selection_key,
    )
    if tuple(entry.semantic_sha256 for entry in entries) != (
        working.artifact_semantic_sha256
    ):
        raise ValueError("Region working-set identity differs from its manifest")
    semantic_library = RegionBlockSemanticLibrary.load(
        block_semantic_manifest_path,
        library,
    )
    snapshots = semantic_library.load_for_region_entries(entries)
    palette_by_key = {}
    for snapshot in snapshots:
        for entry in snapshot.palette:
            previous = palette_by_key.setdefault(entry.semantic_key, entry)
            if previous != entry:
                raise ValueError("Region semantic key has conflicting meanings")

    archive_path = (
        default_hytale_assets_path()
        if assets_path is None
        else Path(assets_path)
    )
    excluded: list[str] = []
    with HytaleAssetArchive(archive_path) as archive:
        resolver = LocalBlockSemanticsResolver(archive)
        compatible = []
        for entry, reference in resolve_region_block_asset_references(
            archive,
            tuple(palette_by_key.values()),
        ):
            item, _ = resolver.resolve_item_asset(reference)
            affordance = resolve_local_block_affordance(
                item,
                item["BlockType"],
            )
            agrees = (
                affordance.valid
                and affordance.tags == entry.affordance_tags
                and affordance.gather_type_index == entry.gather_type_index
                and affordance.required_tool_quality
                == entry.required_tool_quality
                and block_semantic_key(reference, entry.rotation_index)
                == entry.semantic_key
            )
            if agrees:
                compatible.append(entry)
            else:
                excluded.append(reference)
        compiled = compile_region_block_action_catalog(
            archive,
            compatible,
            source_block_semantic_sha256=semantic_library.semantic_sha256,
        )
        item_asset_ids = []
        for row in rows:
            asset_ids = {
                trigger.item_asset_id
                for trigger in row.triggers
                if trigger.item_asset_id
            }
            if len(asset_ids) != 1:
                raise ValueError("native item evidence must identify one item")
            item_asset_ids.append(asset_ids.pop())
        tools = tuple(
            resolve_local_held_item_tool(
                resolver.load_inherited_item_asset(asset_id)[0]
            )
            for asset_id in item_asset_ids
        )
        place_keys, place_valid = _place_semantic_keys(
            rows,
            item_asset_ids,
            resolver,
        )
        state_change_definitions = (
            compile_region_state_change_use_definitions(
                archive,
                compatible,
            )
        )

    interactions = interactions._replace(
        place_semantic_key=jnp.asarray(place_keys, dtype=jnp.uint32),
        place_semantic_key_valid=jnp.asarray(place_valid, dtype=jnp.bool_),
    )

    drop_programs = (
        None
        if not compiled.drop_programs
        else block_drop_program_catalog_from_local(compiled.drop_programs)
    )
    return RegionBlockExecutorAssets(
        interactions=interactions,
        held_tools=RegionHeldToolTable(
            item_id=jnp.asarray(
                [semantic_id(asset_id) for asset_id in item_asset_ids],
                dtype=jnp.int32,
            ),
            valid=jnp.ones((len(item_asset_ids),), dtype=jnp.bool_),
            tool=block_tool_state_from_local(tools),
        ),
        block_catalog=block_action_catalog_from_local(
            compiled.entries,
            drop_programs=compiled.drop_programs,
        ),
        gather_defaults=block_gather_defaults_from_local(
            compiled.gather_defaults
        ),
        drop_programs=drop_programs,
        placed_geometry=compile_region_placed_geometry_table(
            runtime,
            interactions.place_semantic_key,
            interactions.place_semantic_key_valid,
        ),
        state_change_use=(
            None
            if not state_change_definitions
            else compile_region_state_change_use_table(
                runtime,
                state_change_definitions,
            )
        ),
        excluded_asset_references=tuple(sorted(excluded)),
    )


def region_block_interaction_table_from_native_evidence(
    evidence: Sequence[NativeItemInteractionEvidence],
) -> RegionBlockInteractionTable:
    """Compile the unique Primary/Secondary block leaf for each item.

    A trigger with zero or multiple block leaves remains unavailable. This is
    intentionally stricter than root-presence masking: an item is executable
    only when its recorded graph selects one unambiguous Break/Place leaf.
    """

    rows = tuple(evidence)
    if not rows:
        raise ValueError("native item interaction evidence cannot be empty")
    items: list[int] = []
    valid: list[bool] = []
    trigger_valid: list[list[bool]] = []
    programs: list[list[BlockInteractionProgram]] = []
    tool_ids: list[list[int]] = []
    match_tools: list[list[bool]] = []
    seen: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, NativeItemInteractionEvidence):
            raise TypeError("evidence contains a non-native interaction row")
        asset_ids = {
            trigger.item_asset_id
            for trigger in row.triggers
            if trigger.item_asset_id
        }
        if len(asset_ids) != 1:
            raise ValueError("native item evidence must identify one item")
        asset_id = asset_ids.pop()
        item_id = semantic_id(asset_id)
        previous = seen.setdefault(item_id, asset_id)
        if previous != asset_id:
            raise ValueError("native item semantic ID collision")
        item_programs: list[BlockInteractionProgram] = []
        item_trigger_valid: list[bool] = []
        item_tool_ids: list[int] = []
        item_match_tools: list[bool] = []
        for trigger_index in (0, 1):
            trigger = row.triggers[trigger_index]
            leaves = tuple(
                node
                for node in trigger.nodes
                if node.implementation_class.rsplit(".", 1)[-1]
                in {"BreakBlockInteraction", "PlaceBlockInteraction"}
            )
            if len(leaves) != 1:
                item_programs.append(empty_block_interaction_program(1, entity_count=1))
                item_trigger_valid.append(False)
                item_tool_ids.append(0)
                item_match_tools.append(False)
                continue
            node = leaves[0]
            compiled = block_interaction_program_from_native_node(
                node,
                batch_size=1,
                entity_count=1,
                next_program_id=(
                    semantic_id(node.next_interaction_id)
                    if node.next_interaction_id
                    else NO_INTERACTION_PROGRAM
                ),
                failed_program_id=(
                    semantic_id(node.failed_interaction_id)
                    if node.failed_interaction_id
                    else NO_INTERACTION_PROGRAM
                ),
                native_evidence_version=row.version,
            )
            item_programs.append(compiled.program)
            item_trigger_valid.append(True)
            item_tool_ids.append(
                stable_asset_id(compiled.break_tool_id)
                if compiled.break_tool_id
                else 0
            )
            item_match_tools.append(compiled.break_match_tool)
        items.append(item_id)
        valid.append(True)
        trigger_valid.append(item_trigger_valid)
        programs.append(item_programs)
        tool_ids.append(item_tool_ids)
        match_tools.append(item_match_tools)

    def program_field(name: str) -> jax.Array:
        return jnp.asarray(
            [
                [getattr(program, name)[0, 0] for program in item]
                for item in programs
            ]
        )

    return RegionBlockInteractionTable(
        item_id=jnp.asarray(items, dtype=jnp.int32),
        valid=jnp.asarray(valid, dtype=jnp.bool_),
        trigger_valid=jnp.asarray(trigger_valid, dtype=jnp.bool_),
        program=BlockInteractionProgram(
            **{name: program_field(name) for name in BlockInteractionProgram._fields}
        ),
        break_tool_type_id=jnp.asarray(tool_ids, dtype=jnp.int32),
        break_match_tool=jnp.asarray(match_tools, dtype=jnp.bool_),
        place_semantic_key=jnp.zeros(
            (len(items), 2, BLOCK_SEMANTIC_KEY_WORDS),
            dtype=jnp.uint32,
        ),
        place_semantic_key_valid=jnp.zeros(
            (len(items), 2),
            dtype=jnp.bool_,
        ),
    )


def make_region_block_action_surface_executor(
    *,
    block_candidate_provider: ArsenalBlockCandidateProvider,
    interaction_table: RegionBlockInteractionTable,
    held_tool_table: RegionHeldToolTable,
    placed_geometry_table: RegionPlacedGeometryTable,
    block_catalog: BlockActionCatalog,
    gather_defaults: BlockGatherDefaults,
    inventory_layout: InventoryLayout,
    drop_programs: BlockDropProgramCatalog | None = None,
    recipe_candidate_provider=None,
    crafting_assets: RegionFieldcraftAssets | None = None,
    state_change_use_table: RegionStateChangeUseTable | None = None,
    modification_allowed: bool = False,
):
    """Build a JIT-safe Break/Place/fieldcraft executor over one Region."""

    if not callable(block_candidate_provider):
        raise TypeError("block_candidate_provider must be callable")
    if not isinstance(interaction_table, RegionBlockInteractionTable):
        raise TypeError("interaction_table has the wrong type")
    if not isinstance(held_tool_table, RegionHeldToolTable):
        raise TypeError("held_tool_table has the wrong type")
    if not isinstance(placed_geometry_table, RegionPlacedGeometryTable):
        raise TypeError("placed_geometry_table has the wrong type")
    if not isinstance(inventory_layout, InventoryLayout):
        raise TypeError("inventory_layout has the wrong type")
    if (recipe_candidate_provider is None) != (crafting_assets is None):
        raise ValueError(
            "recipe_candidate_provider and crafting_assets must be supplied together"
        )
    if crafting_assets is not None and not isinstance(
        crafting_assets,
        RegionFieldcraftAssets,
    ):
        raise TypeError("crafting_assets has the wrong type")
    if state_change_use_table is not None and not isinstance(
        state_change_use_table,
        RegionStateChangeUseTable,
    ):
        raise TypeError("state_change_use_table has the wrong type")
    permission = bool(modification_allowed)

    def executor(state, surface_runtime, surface, _evidence, keys, params, config):
        if not isinstance(surface_runtime.world, RegionActionRuntimeState):
            raise TypeError("Region action execution requires RegionActionRuntimeState")
        world = surface_runtime.world
        requests = action_surface_verb_requests(surface)
        single_request = (
            jnp.sum(requests.astype(jnp.int32), axis=1) <= jnp.int32(1)
        )
        candidates, _ = block_candidate_provider(state, world, params)
        batch = state.combat.health.shape[0]
        selected_index = surface.block_candidate_index[:, None]
        selection = select_region_block_action_target(
            world.geometry,
            world.semantic_atlas,
            candidates,
            selected_index,
        )

        stack = item_stack_at(
            state.inventory,
            inventory_layout,
            CONTAINER_HOTBAR,
            state.inventory.active_hotbar_slot,
        )
        actor_item = stack.item_id[:, 0]
        trigger = surface.block_interaction_trigger
        trigger_index = jnp.clip(trigger - 1, 0, 1)
        trigger_in_range = (
            (trigger == ARSENAL_BLOCK_TRIGGER_PRIMARY)
            | (trigger == ARSENAL_BLOCK_TRIGGER_SECONDARY)
        )
        program_row, program_unique = _unique_row(
            interaction_table.item_id,
            interaction_table.valid,
            actor_item,
        )
        program_valid = (
            trigger_in_range
            & program_unique
            & _gather_2d(interaction_table.trigger_valid, program_row, trigger_index)
        )
        program = jax.tree.map(
            lambda value: _gather_2d(value, program_row, trigger_index)[:, None],
            interaction_table.program,
        )
        break_selected = (
            program_valid[:, None]
            & (program.kind == BLOCK_INTERACTION_BREAK)
            & (selected_index >= 0)
            & single_request[:, None]
        )
        placements = produce_region_runtime_actor_block_placement_candidates(
            world,
            candidates,
            candidate_capacity=candidates.candidate_mask.shape[2] * 6,
        )
        selected_clicked_position = _gather_actor_candidate(
            candidates.visible_position,
            selected_index,
        )
        selected_face, face_available = select_actor_aim_block_face(
            state,
            params,
            placements,
            selected_clicked_position,
        )
        placed_geometry, placed_geometry_available = (
            select_region_placed_geometry(
                placed_geometry_table,
                world,
                program_row,
                trigger_index,
            )
        )
        place_selected = (
            program_valid[:, None]
            & (program.kind == BLOCK_INTERACTION_PLACE)
            & (selected_index >= 0)
            & face_available[:, None]
            & placed_geometry_available[:, None]
            & single_request[:, None]
        )

        tool_row, tool_unique = _unique_row(
            held_tool_table.item_id,
            held_tool_table.valid,
            actor_item,
        )
        tool = jax.tree.map(
            lambda value: value[tool_row][:, None, ...],
            held_tool_table.tool,
        )
        tool = tool._replace(
            available=tool.available & tool_unique[:, None],
        )
        random_samples = jax.vmap(
            lambda key: jax.random.uniform(key, (2,), dtype=jnp.float32)
        )(keys)[:, None, :]
        known_permission = jnp.full(
            (batch, 1),
            permission,
            dtype=jnp.bool_,
        )
        empty_geometry = empty_mutable_block_geometry((batch, 1))
        world_execution = execute_region_runtime_block_action_with_placement(
            world,
            candidates,
            placements,
            selected_index,
            selected_face[:, None],
            tool,
            gather_defaults,
            block_catalog,
            break_selected=break_selected,
            place_selected=place_selected,
            placed_geometry=placed_geometry,
            replacement_geometry=empty_geometry,
            modification_evidence_available=known_permission,
            modification_allowed=known_permission,
            place_evidence_available=known_permission,
            place_allowed=known_permission,
            break_interaction_tool_type_id=_gather_2d(
                interaction_table.break_tool_type_id,
                program_row,
                trigger_index,
            )[:, None],
            break_interaction_match_tool=_gather_2d(
                interaction_table.break_match_tool,
                program_row,
                trigger_index,
            )[:, None],
            drop_programs=drop_programs,
            drop_random_samples=random_samples,
        )
        use_requested = (
            requests[:, ACTION_SURFACE_VERB_USE, None]
            & (selected_index >= 0)
            & (stack.quantity[:, :1] <= 0)
            & single_request[:, None]
        )
        use_execution = None
        if state_change_use_table is not None:
            use_execution = execute_region_state_change_use(
                world_execution.runtime,
                candidates,
                selected_index,
                state_change_use_table,
                intent_mask=use_requested,
            )
        selection = world_execution.plan.base_selection
        cascade = region_removal_cascade_safety(
            world.semantic_atlas,
            world.geometry.atlas,
            block_catalog,
            selection.position,
            world.geometry.environment_world_id,
        )
        placement_support = region_block_placement_support_result(
            world.geometry.atlas,
            world_execution.plan.placement.destination_position,
            world.geometry.environment_world_id,
        )
        command = BlockInteractionCommand(
            start=break_selected | place_selected,
            target_block=selection.position,
            held_container_id=jnp.full(
                (batch, 1),
                CONTAINER_HOTBAR,
                dtype=jnp.int32,
            ),
            held_container_slot=state.inventory.active_hotbar_slot[:, :1],
        )
        actor_block_state = _actor_block_state(
            surface_runtime.block_interactions
        )
        actor_inventory = _actor_inventory(state.inventory)
        drop_shape = world_execution.plan.actions.break_drops.mask.shape
        pickup_container = jnp.full(
            drop_shape,
            CONTAINER_HOTBAR,
            dtype=jnp.int32,
        )
        hotbar_slots = inventory_layout.container_id == CONTAINER_HOTBAR
        pickup_slots = jnp.broadcast_to(
            hotbar_slots,
            drop_shape + (inventory_layout.container_id.shape[0],),
        )
        next_block, next_inventory, block_info = (
            step_block_interactions_from_world(
                actor_block_state,
                actor_inventory,
                inventory_layout,
                command,
                program,
                world_execution.plan.actions,
                world_execution.acknowledgement,
                creative_mode=jnp.zeros((batch, 1), dtype=jnp.bool_),
                removal_cascade_safe=jnp.where(
                    place_selected,
                    True,
                    cascade.safe,
                ),
                pickup_container_id=pickup_container,
                pickup_slot_allowed=pickup_slots,
                placement_support=placement_support,
                dt_seconds=(jnp.float32(1.0) / params.ticks_per_second),
            )
        )
        was_running = actor_block_state.phase[:, 0] == BLOCK_INTERACTION_RUNNING
        started = block_info.start_accepted[:, 0]
        finished = (started | was_running) & (
            next_block.phase[:, 0] == BLOCK_INTERACTION_SUCCEEDED
        )
        cancelled = block_info.item_change_cancelled[:, 0]
        block_requested = (break_selected | place_selected)[:, 0]
        accepted = jnp.zeros(
            (batch, ACTION_SURFACE_VERB_COUNT),
            dtype=jnp.bool_,
        ).at[:, ACTION_SURFACE_VERB_BLOCK_INTERACTION].set(started)
        start_rows = jnp.zeros_like(accepted).at[
            :, ACTION_SURFACE_VERB_BLOCK_INTERACTION
        ].set(started)
        finish_rows = jnp.zeros_like(accepted).at[
            :, ACTION_SURFACE_VERB_BLOCK_INTERACTION
        ].set(finished)
        cancel_rows = jnp.zeros_like(accepted).at[
            :, ACTION_SURFACE_VERB_BLOCK_INTERACTION
        ].set(cancelled)
        next_world = (
            world_execution.runtime
            if use_execution is None
            else use_execution.runtime
        )
        next_runtime = surface_runtime._replace(
            block_interactions=_merge_actor_block_state(
                surface_runtime.block_interactions,
                next_block,
            ),
            world=next_world,
        )
        next_state = state._replace(
            inventory=_merge_actor_inventory(
                state.inventory,
                next_inventory,
            )
        )
        use_rechecked = ~requests[:, ACTION_SURFACE_VERB_USE]
        use_accepted = jnp.zeros((batch,), dtype=jnp.bool_)
        if use_execution is not None:
            use_accepted = use_execution.accepted[:, 0]
            use_rechecked = (
                ~requests[:, ACTION_SURFACE_VERB_USE]
                | use_execution.selection.available[:, 0]
            )
            accepted = accepted.at[:, ACTION_SURFACE_VERB_USE].set(
                use_accepted
            )
            start_rows = start_rows.at[:, ACTION_SURFACE_VERB_USE].set(
                use_accepted
            )
            finish_rows = finish_rows.at[:, ACTION_SURFACE_VERB_USE].set(
                use_accepted
            )
        craft_rechecked = ~requests[:, ACTION_SURFACE_VERB_CRAFT]
        craft_accepted = jnp.zeros((batch,), dtype=jnp.bool_)
        if crafting_assets is not None:
            recipe_candidates = recipe_candidate_provider(
                next_state,
                next_world,
                params,
            )
            craft = execute_region_fieldcraft(
                crafting_assets,
                next_state.inventory,
                recipe_candidates,
                jnp.where(
                    single_request,
                    surface.recipe_candidate_index,
                    jnp.int32(-1),
                ),
            )
            craft_rechecked = craft.target_rechecked
            craft_accepted = craft.accepted
            next_state = next_state._replace(inventory=craft.state)
            accepted = accepted.at[:, ACTION_SURFACE_VERB_CRAFT].set(
                craft_accepted
            )
            start_rows = start_rows.at[:, ACTION_SURFACE_VERB_CRAFT].set(
                craft_accepted
            )
            finish_rows = finish_rows.at[:, ACTION_SURFACE_VERB_CRAFT].set(
                craft.finished
            )
        any_request = jnp.any(requests, axis=1)
        all_requested_accepted = jnp.all(~requests | accepted, axis=1)
        request_legal = ~any_request | (
            single_request & all_requested_accepted
        )
        block_rechecked = ~requests[:, ACTION_SURFACE_VERB_BLOCK_INTERACTION] | (
            block_requested & selection.available[:, 0]
        )
        target_rechecked = block_rechecked & craft_rechecked & use_rechecked
        state_commit = started | was_running | craft_accepted | use_accepted
        return ArsenalActionSurfaceExecution(
            state=next_state,
            runtime=next_runtime,
            request_legal=request_legal,
            selected_target_rechecked=target_rechecked,
            state_commit=state_commit,
            lifecycle=action_surface_lifecycle_evidence(
                surface,
                accepted=accepted,
                started=start_rows,
                finished=finish_rows,
                cancelled=cancel_rows,
                reject_reason=jnp.full(
                    (batch, ACTION_SURFACE_VERB_COUNT),
                    ACTION_SURFACE_REJECT_EXECUTOR_DENIED,
                    dtype=jnp.uint32,
                ),
            ),
        )

    executor.contract_sha256 = region_block_executor_contract_sha256()
    executor.block_candidate_provider = block_candidate_provider
    executor.placed_geometry_table = placed_geometry_table
    executor.recipe_candidate_provider = recipe_candidate_provider
    executor.crafting_assets = crafting_assets
    executor.state_change_use_table = state_change_use_table
    return executor


def _unique_row(item_id, valid, requested):
    matches = valid[None, :] & (item_id[None, :] == requested[:, None])
    unique = jnp.sum(matches.astype(jnp.int32), axis=1) == 1
    return jnp.argmax(matches, axis=1).astype(jnp.int32), unique


def _place_semantic_keys(
    evidence: Sequence[NativeItemInteractionEvidence],
    item_asset_ids: Sequence[str],
    resolver: LocalBlockSemanticsResolver,
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.zeros(
        (len(evidence), 2, BLOCK_SEMANTIC_KEY_WORDS),
        dtype=np.uint32,
    )
    valid = np.zeros((len(evidence), 2), dtype=np.bool_)
    for row_index, (row, item_asset_id) in enumerate(
        zip(evidence, item_asset_ids, strict=True)
    ):
        item, _ = resolver.load_inherited_item_asset(item_asset_id)
        held_block = (
            item_asset_id
            if isinstance(item.get("BlockType"), dict)
            else None
        )
        for trigger_index in (0, 1):
            leaves = tuple(
                node
                for node in row.triggers[trigger_index].nodes
                if node.implementation_class.rsplit(".", 1)[-1]
                == "PlaceBlockInteraction"
            )
            if len(leaves) != 1:
                continue
            payload = leaves[0].payload
            if not isinstance(payload, NativePlaceBlockPayload):
                continue
            reference = payload.block_asset_id or held_block
            if reference is None:
                continue
            keys[row_index, trigger_index] = np.asarray(
                block_semantic_key(reference, 0),
                dtype=np.uint32,
            )
            valid[row_index, trigger_index] = True
    return keys, valid


def _gather_2d(value, row, column):
    return value[row, column]


def _gather_actor_candidate(value, index):
    """Gather one bounded candidate per actor without host indexing."""

    array = jnp.asarray(value)
    raw_index = jnp.asarray(index, dtype=jnp.int32)
    if array.ndim < 3 or array.shape[:2] != raw_index.shape:
        raise ValueError("candidate value must have shape [batch, actor, ...]")
    safe_index = jnp.clip(raw_index, 0, array.shape[2] - 1)
    gather = safe_index[..., None].reshape(
        safe_index.shape + (1,) * (array.ndim - safe_index.ndim)
    )
    gather = jnp.broadcast_to(
        gather,
        array.shape[:2] + (1,) + array.shape[3:],
    )
    selected = jnp.take_along_axis(array, gather, axis=2)[:, :, 0]
    valid = (raw_index >= 0) & (raw_index < array.shape[2])
    gate = valid.reshape(valid.shape + (1,) * (selected.ndim - valid.ndim))
    return jnp.where(gate, selected, jnp.zeros_like(selected))


def _actor_inventory(state: InventoryState) -> InventoryState:
    return InventoryState(
        **{
            name: (
                value
                if name == "failure_bits"
                else value[:, :1, ...]
            )
            for name, value in state._asdict().items()
        }
    )


def _merge_actor_inventory(full: InventoryState, actor: InventoryState) -> InventoryState:
    return InventoryState(
        **{
            name: (
                actor.failure_bits
                if name == "failure_bits"
                else value.at[:, 0, ...].set(getattr(actor, name)[:, 0, ...])
            )
            for name, value in full._asdict().items()
        }
    )


def _actor_block_state(state: BlockInteractionState) -> BlockInteractionState:
    return BlockInteractionState(
        **{
            name: (
                value
                if name == "failure_bits"
                else value[:, :1, ...]
            )
            for name, value in state._asdict().items()
        }
    )


def _merge_actor_block_state(
    full: BlockInteractionState,
    actor: BlockInteractionState,
) -> BlockInteractionState:
    return BlockInteractionState(
        **{
            name: (
                actor.failure_bits
                if name == "failure_bits"
                else value.at[:, 0, ...].set(getattr(actor, name)[:, 0, ...])
            )
            for name, value in full._asdict().items()
        }
    )


__all__ = [
    "RegionBlockExecutorAssets",
    "RegionBlockInteractionTable",
    "RegionHeldToolTable",
    "REGION_BLOCK_EXECUTOR_SCHEMA",
    "REGION_BLOCK_EXECUTOR_VERSION",
    "compile_region_block_executor_assets",
    "make_region_block_action_surface_executor",
    "make_region_block_item_availability_provider",
    "make_region_held_item_inventory_reset_provider",
    "region_block_interaction_table_from_native_evidence",
    "region_block_executor_contract_sha256",
]
