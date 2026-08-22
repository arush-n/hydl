"""Action catalog assembly and drop program resolution."""
from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.mutable_blocks import (
    BLOCK_SEMANTIC_KEY_WORDS,
)
from hytalegym.worldgen.block_actions import (
    BLOCK_DROP_METADATA_HASH_WORDS,
    BLOCK_DROP_OUTPUT_CAPACITY,
    LocalBlockActionSemantics,
    SUPPORT_DROP_FALL,
    SUPPORT_DROP_NONE,
)
from hytalegym.worldgen.block_affordances import (
    block_affordance_tag_mask,
)
from hytalegym.worldgen.drop_programs import (
    BLOCK_DROP_PROGRAM_CAPACITY,
    LocalDropProgram,
)

from hytalegym.jax.world.block_actions._catalog import (
    BlockActionCatalog,
    BlockActionTargetResult,
    _device_drop_table,
    _validate_drop_program_catalog,
)
from hytalegym.jax.world.block_actions._primitives import (
    BlockDropProgramCatalog,
    BlockDropTable,
    _capacity,
    _host_drop_table,
    _install_drop_route,
)


Array = jax.Array
BLOCK_ACTION_TARGET_SCHEMA = "hytalerl_block_action_target_v4"
BLOCK_ACTION_TARGET_VERSION = 4

BLOCK_ACTION_DIAGNOSTIC_INVALID = 1 << 0
BLOCK_ACTION_DIAGNOSTIC_TARGET_UNAVAILABLE = 1 << 1
BLOCK_ACTION_DIAGNOSTIC_CATALOG_MISSING = 1 << 2
BLOCK_ACTION_DIAGNOSTIC_CATALOG_DUPLICATE = 1 << 3
BLOCK_ACTION_DIAGNOSTIC_TOOL_UNAVAILABLE = 1 << 4
BLOCK_ACTION_DIAGNOSTIC_DROP_UNAVAILABLE = 1 << 5
BLOCK_ACTION_DIAGNOSTIC_PERMISSION_UNAVAILABLE = 1 << 6
BLOCK_ACTION_DIAGNOSTIC_PLACE_UNAVAILABLE = 1 << 7
BLOCK_ACTION_DIAGNOSTIC_INTERACTION_TOOL_MISMATCH = 1 << 8

BLOCK_ACTION_MUTATION_SCHEMA = "hytalerl_block_action_mutation_v1"
BLOCK_ACTION_MUTATION_VERSION = 1
BLOCK_ACTION_MUTATION_DIAGNOSTIC_SELECTION = 1 << 0
BLOCK_ACTION_MUTATION_DIAGNOSTIC_PREDICATE = 1 << 1
BLOCK_ACTION_MUTATION_DIAGNOSTIC_GEOMETRY = 1 << 2
BLOCK_ACTION_MUTATION_DIAGNOSTIC_PROVENANCE = 1 << 3

_TAG_BREAKABLE = np.uint16(block_affordance_tag_mask("breakable"))
_TAG_HARVESTABLE = np.uint16(block_affordance_tag_mask("harvestable"))
_TAG_SOFT = np.uint16(block_affordance_tag_mask("soft"))


def _resolve_drop_program(
    drops: BlockDropTable,
    programs: BlockDropProgramCatalog | None,
    sample: Array,
    *,
    samples_supplied: bool,
) -> BlockDropTable:
    program_index = drops.program_index.astype(jnp.int32)
    uses_program = program_index >= 0
    if programs is None:
        available = drops.available & ~uses_program
        gate = ~uses_program[..., None]
        return drops._replace(
            available=available,
            mask=drops.mask & gate,
            item_id=jnp.where(gate, drops.item_id, 0),
            quantity=jnp.where(gate, drops.quantity, 0),
            item_max_stack=jnp.where(gate, drops.item_max_stack, 0),
            durability=jnp.where(gate, drops.durability, 0.0),
            max_durability=jnp.where(
                gate,
                drops.max_durability,
                0.0,
            ),
            metadata_hash=jnp.where(
                gate[..., None],
                drops.metadata_hash,
                0,
            ),
        )
    _validate_drop_program_catalog(programs)
    capacity = programs.program_mask.shape[0]
    safe_program = jnp.clip(program_index, 0, capacity - 1)
    program_known = uses_program & programs.program_mask[safe_program]
    random_required = programs.randomized[safe_program]
    sample_valid = (
        jnp.isfinite(sample)
        & (sample >= 0.0)
        & (sample < 1.0)
    )
    random_available = ~random_required | (
        samples_supplied & sample_valid
    )
    candidates = (
        programs.outcome_mask[safe_program]
        & (
            sample[..., None]
            < programs.cumulative_probability[safe_program]
        )
    )
    outcome_known = jnp.any(candidates, axis=-1)
    outcome_index = jnp.argmax(candidates, axis=-1).astype(jnp.int32)
    program_available = (
        program_known & random_available & outcome_known
    )
    program_drop_mask = programs.drop_mask[
        safe_program,
        outcome_index,
    ]
    program_item_id = programs.item_id[safe_program, outcome_index]
    program_quantity = programs.quantity[safe_program, outcome_index]
    program_max_stack = programs.item_max_stack[
        safe_program,
        outcome_index,
    ]
    program_durability = programs.durability[
        safe_program,
        outcome_index,
    ]
    program_max_durability = programs.max_durability[
        safe_program,
        outcome_index,
    ]
    program_metadata = programs.metadata_hash[
        safe_program,
        outcome_index,
    ]
    program_gate = uses_program[..., None]
    available = drops.available & (
        ~uses_program | program_available
    )
    output_gate = program_available[..., None]
    return BlockDropTable(
        available=available,
        program_index=drops.program_index,
        mask=jnp.where(
            program_gate,
            output_gate & program_drop_mask,
            drops.mask,
        ),
        item_id=jnp.where(
            program_gate,
            jnp.where(output_gate, program_item_id, 0),
            drops.item_id,
        ),
        quantity=jnp.where(
            program_gate,
            jnp.where(output_gate, program_quantity, 0),
            drops.quantity,
        ),
        item_max_stack=jnp.where(
            program_gate,
            jnp.where(output_gate, program_max_stack, 0),
            drops.item_max_stack,
        ),
        durability=jnp.where(
            program_gate,
            jnp.where(output_gate, program_durability, 0.0),
            drops.durability,
        ),
        max_durability=jnp.where(
            program_gate,
            jnp.where(output_gate, program_max_durability, 0.0),
            drops.max_durability,
        ),
        metadata_hash=jnp.where(
            program_gate[..., None],
            jnp.where(
                output_gate[..., None],
                program_metadata,
                0,
            ),
            drops.metadata_hash,
        ),
    )


def _validate_action_result(
    actions: BlockActionTargetResult,
    shape: tuple[int, ...],
) -> None:
    for label, value in (
        ("break_allowed", actions.break_allowed),
        ("break_removes_block", actions.break_removes_block),
        (
            "replacement_semantic_key_valid",
            actions.replacement_semantic_key_valid,
        ),
        ("place_allowed", actions.place_allowed),
    ):
        if value.shape != shape:
            raise ValueError(f"actions.{label} has the wrong shape")
    if actions.replacement_semantic_key.shape != shape + (
        BLOCK_SEMANTIC_KEY_WORDS,
    ):
        raise ValueError(
            "actions.replacement_semantic_key has the wrong shape"
        )


def _validate_catalog(catalog: BlockActionCatalog) -> None:
    if not isinstance(catalog, BlockActionCatalog):
        raise TypeError("catalog must be BlockActionCatalog")
    capacity = catalog.entry_mask.shape[0]
    if catalog.semantic_key.shape != (
        capacity,
        BLOCK_SEMANTIC_KEY_WORDS,
    ):
        raise ValueError("catalog semantic keys have the wrong shape")
    for name in (
        "support_dependent",
        "support_drop_type",
        "tool_breakable",
        "soft_breakable",
        "soft_weapon_breakable",
        "harvestable",
        "interaction_tool_route_present",
        "interaction_tool_route_type_id",
        "interaction_tool_replacement_valid",
    ):
        if getattr(catalog, name).shape != (capacity,):
            raise ValueError(f"catalog.{name} has the wrong shape")
    if catalog.interaction_tool_replacement_semantic_key.shape != (
        capacity,
        BLOCK_SEMANTIC_KEY_WORDS,
    ):
        raise ValueError(
            "catalog interaction replacement keys have the wrong shape"
        )
    for name in (
        "interaction_tool_drop",
        "tool_drop",
        "soft_drop",
        "harvest_drop",
    ):
        table = getattr(catalog, name)
        drops = table.mask.shape[1]
        if not 1 <= drops <= BLOCK_DROP_OUTPUT_CAPACITY:
            raise ValueError("catalog drop capacity is invalid")
        if table.available.shape != (capacity,):
            raise ValueError(f"catalog.{name}.available has the wrong shape")
        if table.program_index.shape != (capacity,):
            raise ValueError(
                f"catalog.{name}.program_index has the wrong shape"
            )
        for field in BlockDropTable._fields[2:]:
            expected = (
                (capacity, drops, BLOCK_DROP_METADATA_HASH_WORDS)
                if field == "metadata_hash"
                else (capacity, drops)
            )
            if getattr(table, field).shape != expected:
                raise ValueError(f"catalog.{name}.{field} has the wrong shape")


def block_action_catalog_from_local(
    entries: Sequence[LocalBlockActionSemantics],
    *,
    drop_programs: Sequence[LocalDropProgram] = (),
    entry_capacity: int | None = None,
    drop_capacity: int = BLOCK_DROP_OUTPUT_CAPACITY,
) -> BlockActionCatalog:
    """Pad source-resolved block routes and reject semantic-key aliases."""

    source = tuple(entries)
    if not source:
        raise ValueError("block action catalog needs at least one entry")
    if any(not isinstance(item, LocalBlockActionSemantics) for item in source):
        raise TypeError("catalog entries must be LocalBlockActionSemantics")
    capacity = len(source) if entry_capacity is None else entry_capacity
    capacity = _capacity(capacity, 65_535, "block catalog")
    if capacity < len(source):
        raise ValueError("block catalog capacity is smaller than its entries")
    drops = _capacity(
        drop_capacity,
        BLOCK_DROP_OUTPUT_CAPACITY,
        "drop output",
    )
    keys = np.zeros(
        (capacity, BLOCK_SEMANTIC_KEY_WORDS),
        dtype=np.uint32,
    )
    mask = np.zeros(capacity, dtype=np.bool_)
    tool = np.zeros(capacity, dtype=np.bool_)
    support_dependent = np.zeros(capacity, dtype=np.bool_)
    support_drop_type = np.zeros(capacity, dtype=np.uint8)
    soft = np.zeros(capacity, dtype=np.bool_)
    soft_weapon = np.zeros(capacity, dtype=np.bool_)
    harvest = np.zeros(capacity, dtype=np.bool_)
    interaction_route = np.zeros(capacity, dtype=np.bool_)
    interaction_route_type = np.zeros(capacity, dtype=np.int32)
    interaction_replacement_valid = np.zeros(capacity, dtype=np.bool_)
    interaction_replacement_key = np.zeros(
        (capacity, BLOCK_SEMANTIC_KEY_WORDS),
        dtype=np.uint32,
    )
    route_tables = {
        name: _host_drop_table(capacity, drops)
        for name in (
            "interaction_tool_drop",
            "tool_drop",
            "soft_drop",
            "harvest_drop",
        )
    }
    program_source = tuple(drop_programs)
    if any(not isinstance(item, LocalDropProgram) for item in program_source):
        raise TypeError("drop programs must contain LocalDropProgram values")
    if len(program_source) > BLOCK_DROP_PROGRAM_CAPACITY:
        raise ValueError("drop-program capacity exceeded")
    program_indices: dict[str, int] = {}
    for index, program in enumerate(program_source):
        if program.semantic_sha256 in program_indices:
            raise ValueError("drop programs have duplicate semantic identities")
        program_indices[program.semantic_sha256] = index
    seen: set[tuple[int, ...]] = set()
    for index, entry in enumerate(source):
        if entry.semantic_key in seen:
            raise ValueError("block action catalog has duplicate semantic keys")
        if not isinstance(entry.support_dependent, bool):
            raise TypeError("support_dependent must be boolean")
        if (
            isinstance(entry.support_drop_type, bool)
            or not isinstance(entry.support_drop_type, int)
            or not SUPPORT_DROP_NONE
            <= entry.support_drop_type
            <= SUPPORT_DROP_FALL
            or (
                entry.support_dependent
                != (entry.support_drop_type != SUPPORT_DROP_NONE)
            )
        ):
            raise ValueError("support drop semantics are inconsistent")
        seen.add(entry.semantic_key)
        keys[index] = np.asarray(entry.semantic_key, dtype=np.uint32)
        mask[index] = True
        support_dependent[index] = entry.support_dependent
        support_drop_type[index] = entry.support_drop_type
        tool[index] = entry.tool_breakable
        soft[index] = entry.soft_breakable
        soft_weapon[index] = entry.soft_weapon_breakable
        harvest[index] = entry.harvestable
        authored_route = entry.interaction_tool_route
        interaction_route[index] = authored_route.present
        interaction_route_type[index] = authored_route.tool_type_id
        interaction_replacement_valid[
            index
        ] = authored_route.replacement_semantic_key_valid
        interaction_replacement_key[index] = np.asarray(
            authored_route.replacement_semantic_key,
            dtype=np.uint32,
        )
        _install_drop_route(
            route_tables["interaction_tool_drop"],
            index,
            authored_route.drop,
            program_indices,
        )
        for name in ("tool_drop", "soft_drop", "harvest_drop"):
            _install_drop_route(
                route_tables[name],
                index,
                getattr(entry, name),
                program_indices,
            )
    return BlockActionCatalog(
        entry_mask=jnp.asarray(mask),
        semantic_key=jnp.asarray(keys),
        support_dependent=jnp.asarray(support_dependent),
        support_drop_type=jnp.asarray(support_drop_type),
        tool_breakable=jnp.asarray(tool),
        soft_breakable=jnp.asarray(soft),
        soft_weapon_breakable=jnp.asarray(soft_weapon),
        harvestable=jnp.asarray(harvest),
        interaction_tool_route_present=jnp.asarray(interaction_route),
        interaction_tool_route_type_id=jnp.asarray(interaction_route_type),
        interaction_tool_replacement_valid=jnp.asarray(
            interaction_replacement_valid
        ),
        interaction_tool_replacement_semantic_key=jnp.asarray(
            interaction_replacement_key
        ),
        interaction_tool_drop=_device_drop_table(
            route_tables["interaction_tool_drop"]
        ),
        tool_drop=_device_drop_table(route_tables["tool_drop"]),
        soft_drop=_device_drop_table(route_tables["soft_drop"]),
        harvest_drop=_device_drop_table(route_tables["harvest_drop"]),
    )
