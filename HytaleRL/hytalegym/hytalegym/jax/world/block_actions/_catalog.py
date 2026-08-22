"""Drop program, tool state, and gather default catalog construction."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from fractions import Fraction
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.mutable_blocks import (
    BLOCK_SEMANTIC_KEY_WORDS,
    MutableBlockUpdateResult,
)
from hytalegym.worldgen.block_actions import (
    BLOCK_DROP_METADATA_HASH_WORDS,
    BLOCK_DROP_OUTPUT_CAPACITY,
    BLOCK_TOOL_SPEC_CAPACITY,
    LocalHeldItemTool,
    LocalToolSpec,
)
from hytalegym.worldgen.block_affordances import (
    GATHER_TYPES,
    block_affordance_tag_mask,
)
from hytalegym.worldgen.drop_programs import (
    BLOCK_DROP_PROGRAM_CAPACITY,
    BLOCK_DROP_PROGRAM_OUTCOME_CAPACITY,
    LocalDropProgram,
)

from hytalegym.jax.world.block_actions._primitives import (
    BlockDropProgramCatalog,
    BlockDropTable,
    BlockGatherDefaults,
    BlockMutationAcknowledgement,
    BlockResolvedDrops,
    BlockToolState,
    NativeBlockInteractionOutcome,
    _capacity,
    _prefix,
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


class BlockActionCatalog(NamedTuple):
    """Portable semantic-key lookup, independent of runtime block ordinals."""

    entry_mask: Array
    semantic_key: Array
    support_dependent: Array
    support_drop_type: Array
    tool_breakable: Array
    soft_breakable: Array
    soft_weapon_breakable: Array
    harvestable: Array
    interaction_tool_route_present: Array
    interaction_tool_route_type_id: Array
    interaction_tool_replacement_valid: Array
    interaction_tool_replacement_semantic_key: Array
    interaction_tool_drop: BlockDropTable
    tool_drop: BlockDropTable
    soft_drop: BlockDropTable
    harvest_drop: BlockDropTable


class BlockActionTargetResult(NamedTuple):
    """Action predicates and deterministic outputs for exact target cells."""

    target_available: Array
    diagnostics: Array
    catalog_index: Array
    break_available: Array
    break_allowed: Array
    tool_compatible: Array
    tool_power: Array
    interaction_tool_route_matched: Array
    break_removes_block: Array
    replacement_semantic_key_valid: Array
    replacement_semantic_key: Array
    harvest_available: Array
    harvest_allowed: Array
    target_harvestable: Array
    place_available: Array
    place_allowed: Array
    break_drops: BlockResolvedDrops
    harvest_drops: BlockResolvedDrops


def _device_drop_table(table: dict[str, np.ndarray]) -> BlockDropTable:
    return BlockDropTable(*(jnp.asarray(table[name]) for name in BlockDropTable._fields))


def _gather_drop_table(table: BlockDropTable, slot: Array) -> BlockDropTable:
    return BlockDropTable(*(value[slot] for value in table))


def _publish_drops(
    action_allowed: Array,
    drops: BlockDropTable,
) -> BlockResolvedDrops:
    gate = action_allowed[..., None]
    return BlockResolvedDrops(
        available=action_allowed & drops.available,
        mask=gate & drops.mask,
        item_id=jnp.where(gate, drops.item_id, 0),
        quantity=jnp.where(gate, drops.quantity, 0),
        item_max_stack=jnp.where(gate, drops.item_max_stack, 0),
        durability=jnp.where(gate, drops.durability, 0.0),
        max_durability=jnp.where(gate, drops.max_durability, 0.0),
        metadata_hash=jnp.where(gate[..., None], drops.metadata_hash, 0),
    )


def _select_drop_table(
    condition: Array,
    first: BlockDropTable,
    second: BlockDropTable,
) -> BlockDropTable:
    return BlockDropTable(
        *(
            jnp.where(
                condition.reshape(
                    condition.shape
                    + (1,) * (first_value.ndim - condition.ndim)
                ),
                first_value,
                second_value,
            )
            for first_value, second_value in zip(first, second)
        )
    )


def _tool_value_valid(tool: BlockToolState) -> Array:
    """Validate traced tool evidence without trusting its producer."""

    gather = tool.gather_type_index.astype(jnp.int32)
    active = tool.spec_mask
    active_valid = (
        (gather >= 0)
        & (gather < len(GATHER_TYPES))
        & (tool.quality >= 0)
        & jnp.isfinite(tool.power)
        & (tool.power > 0.0)
    )
    inactive_canonical = (
        (gather == 0)
        & (tool.quality == 0)
        & (tool.power == 0.0)
    )
    counts = jnp.sum(
        active[..., :, None]
        & (
            gather[..., :, None]
            == jnp.arange(len(GATHER_TYPES), dtype=jnp.int32)
        ),
        axis=-2,
    )
    return (
        jnp.all(jnp.where(active, active_valid, inactive_canonical), axis=-1)
        & ~jnp.any(counts > 1, axis=-1)
        & (tool.tool_present | ~jnp.any(active, axis=-1))
    )


def _validate_defaults(defaults: BlockGatherDefaults) -> None:
    if not isinstance(defaults, BlockGatherDefaults):
        raise TypeError("defaults must be BlockGatherDefaults")
    shape = (len(GATHER_TYPES),)
    if any(value.shape != shape for value in defaults):
        raise ValueError("gather defaults differ from the gather dictionary")


def _validate_drop_program_catalog(
    catalog: BlockDropProgramCatalog,
) -> None:
    if not isinstance(catalog, BlockDropProgramCatalog):
        raise TypeError("drop_programs must be BlockDropProgramCatalog")
    capacity = catalog.program_mask.shape[0]
    if not 1 <= capacity <= BLOCK_DROP_PROGRAM_CAPACITY:
        raise ValueError("drop-program catalog capacity is invalid")
    if catalog.semantic_sha256.shape != (
        capacity,
        BLOCK_SEMANTIC_KEY_WORDS,
    ):
        raise ValueError("drop-program semantic identities have wrong shape")
    if catalog.randomized.shape != (capacity,):
        raise ValueError("drop-program randomized flags have wrong shape")
    outcomes = catalog.outcome_mask.shape[1]
    if not 1 <= outcomes <= BLOCK_DROP_PROGRAM_OUTCOME_CAPACITY:
        raise ValueError("drop-program outcome capacity is invalid")
    for name in ("outcome_mask", "cumulative_probability"):
        if getattr(catalog, name).shape != (capacity, outcomes):
            raise ValueError(f"drop_programs.{name} has the wrong shape")
    drops = catalog.drop_mask.shape[2]
    if not 1 <= drops <= BLOCK_DROP_OUTPUT_CAPACITY:
        raise ValueError("drop-program output capacity is invalid")
    for name in (
        "drop_mask",
        "item_id",
        "quantity",
        "item_max_stack",
        "durability",
        "max_durability",
    ):
        if getattr(catalog, name).shape != (capacity, outcomes, drops):
            raise ValueError(f"drop_programs.{name} has the wrong shape")
    if catalog.metadata_hash.shape != (
        capacity,
        outcomes,
        drops,
        BLOCK_DROP_METADATA_HASH_WORDS,
    ):
        raise ValueError("drop_programs.metadata_hash has the wrong shape")


def _validate_tool(tool: BlockToolState, shape: tuple[int, ...]) -> None:
    if not isinstance(tool, BlockToolState):
        raise TypeError("tool must be BlockToolState")
    for name in ("available", "weapon", "builder_tool", "tool_present"):
        if getattr(tool, name).shape != shape:
            raise ValueError(f"tool.{name} must have shape {shape}")
    capacity = tool.spec_mask.shape[-1]
    if not 1 <= capacity <= BLOCK_TOOL_SPEC_CAPACITY:
        raise ValueError("tool spec capacity is invalid")
    for name in ("spec_mask", "gather_type_index", "quality", "power"):
        if getattr(tool, name).shape != shape + (capacity,):
            raise ValueError(
                f"tool.{name} must have shape {shape + (capacity,)}"
            )


def block_drop_program_catalog_from_local(
    programs: Sequence[LocalDropProgram],
    *,
    program_capacity: int | None = None,
    outcome_capacity: int = BLOCK_DROP_PROGRAM_OUTCOME_CAPACITY,
    drop_capacity: int = BLOCK_DROP_OUTPUT_CAPACITY,
) -> BlockDropProgramCatalog:
    """Pack exact host distributions into one bounded device catalog."""

    source = tuple(programs)
    if not source:
        raise ValueError("drop-program catalog cannot be empty")
    if any(not isinstance(item, LocalDropProgram) for item in source):
        raise TypeError("drop-program catalog has the wrong value type")
    capacity = len(source) if program_capacity is None else program_capacity
    capacity = _capacity(capacity, BLOCK_DROP_PROGRAM_CAPACITY, "drop program")
    if capacity < len(source):
        raise ValueError("drop-program capacity is smaller than its entries")
    outcomes = _capacity(
        outcome_capacity,
        BLOCK_DROP_PROGRAM_OUTCOME_CAPACITY,
        "drop-program outcome",
    )
    drops = _capacity(
        drop_capacity,
        BLOCK_DROP_OUTPUT_CAPACITY,
        "drop output",
    )
    program_mask = np.zeros(capacity, dtype=np.bool_)
    semantic = np.zeros(
        (capacity, BLOCK_SEMANTIC_KEY_WORDS),
        dtype=np.uint32,
    )
    randomized = np.zeros(capacity, dtype=np.bool_)
    outcome_mask = np.zeros((capacity, outcomes), dtype=np.bool_)
    cumulative = np.zeros((capacity, outcomes), dtype=np.float32)
    drop_mask = np.zeros((capacity, outcomes, drops), dtype=np.bool_)
    item_id = np.zeros((capacity, outcomes, drops), dtype=np.int32)
    quantity = np.zeros((capacity, outcomes, drops), dtype=np.int32)
    maximum_stack = np.zeros((capacity, outcomes, drops), dtype=np.int32)
    durability = np.zeros((capacity, outcomes, drops), dtype=np.float32)
    maximum_durability = np.zeros(
        (capacity, outcomes, drops),
        dtype=np.float32,
    )
    metadata = np.zeros(
        (
            capacity,
            outcomes,
            drops,
            BLOCK_DROP_METADATA_HASH_WORDS,
        ),
        dtype=np.uint32,
    )
    seen: set[str] = set()
    for program_index, program in enumerate(source):
        if program.semantic_sha256 in seen:
            raise ValueError("drop-program semantic identities must be unique")
        seen.add(program.semantic_sha256)
        if len(program.outcomes) > outcomes:
            raise ValueError("drop program exceeds outcome capacity")
        program_mask[program_index] = True
        semantic[program_index] = np.frombuffer(
            bytes.fromhex(program.semantic_sha256),
            dtype=">u4",
        ).astype(np.uint32)
        randomized[program_index] = program.randomized
        running = Fraction(0)
        for outcome_index, outcome in enumerate(program.outcomes):
            if len(outcome.drops) > drops:
                raise ValueError("drop program exceeds output capacity")
            probability = Fraction(
                outcome.probability_numerator,
                outcome.probability_denominator,
            )
            if probability <= 0:
                raise ValueError("drop outcome probability must be positive")
            running += probability
            outcome_mask[program_index, outcome_index] = True
            cumulative[program_index, outcome_index] = float(running)
            for drop_index, drop in enumerate(outcome.drops):
                drop_mask[
                    program_index,
                    outcome_index,
                    drop_index,
                ] = True
                item_id[program_index, outcome_index, drop_index] = drop.item_id
                quantity[
                    program_index,
                    outcome_index,
                    drop_index,
                ] = drop.quantity
                maximum_stack[
                    program_index,
                    outcome_index,
                    drop_index,
                ] = drop.item_max_stack
                durability[
                    program_index,
                    outcome_index,
                    drop_index,
                ] = drop.durability
                maximum_durability[
                    program_index,
                    outcome_index,
                    drop_index,
                ] = drop.max_durability
                metadata[
                    program_index,
                    outcome_index,
                    drop_index,
                ] = np.asarray(drop.metadata_hash, dtype=np.uint32)
        if running != 1:
            raise ValueError("drop program probabilities must sum to one")
        cumulative[program_index, len(program.outcomes) - 1] = 1.0
    return BlockDropProgramCatalog(
        program_mask=jnp.asarray(program_mask),
        semantic_sha256=jnp.asarray(semantic),
        randomized=jnp.asarray(randomized),
        outcome_mask=jnp.asarray(outcome_mask),
        cumulative_probability=jnp.asarray(cumulative),
        drop_mask=jnp.asarray(drop_mask),
        item_id=jnp.asarray(item_id),
        quantity=jnp.asarray(quantity),
        item_max_stack=jnp.asarray(maximum_stack),
        durability=jnp.asarray(durability),
        max_durability=jnp.asarray(maximum_durability),
        metadata_hash=jnp.asarray(metadata),
    )


def block_gather_defaults_from_local(
    specs: Iterable[LocalToolSpec | None],
) -> BlockGatherDefaults:
    """Build the exact fixed gather dictionary outside JIT."""

    source = tuple(specs)
    if len(source) != len(GATHER_TYPES):
        raise ValueError("gather defaults must match the gather dictionary")
    available = np.zeros(len(source), dtype=np.bool_)
    quality = np.zeros(len(source), dtype=np.int16)
    power = np.zeros(len(source), dtype=np.float32)
    for index, spec in enumerate(source):
        if spec is None:
            continue
        if not isinstance(spec, LocalToolSpec):
            raise TypeError("gather defaults must contain LocalToolSpec or None")
        if spec.gather_type_index != index:
            raise ValueError("gather default index differs from its asset ID")
        available[index] = True
        quality[index] = spec.quality
        power[index] = spec.power
    return BlockGatherDefaults(
        available=jnp.asarray(available),
        quality=jnp.asarray(quality),
        power=jnp.asarray(power),
    )


def block_tool_state_from_local(
    items: Iterable[LocalHeldItemTool],
    *,
    prefix_shape: tuple[int, ...] | None = None,
    spec_capacity: int = BLOCK_TOOL_SPEC_CAPACITY,
) -> BlockToolState:
    """Pack source-resolved held tools into a fixed-shape device value."""

    source = tuple(items)
    if not source:
        raise ValueError("held-tool source cannot be empty")
    prefix = (
        (len(source),)
        if prefix_shape is None
        else _prefix(prefix_shape)
    )
    if math.prod(prefix) != len(source):
        raise ValueError("prefix_shape does not match held-tool count")
    capacity = _capacity(
        spec_capacity,
        BLOCK_TOOL_SPEC_CAPACITY,
        "tool spec",
    )
    available = np.zeros(len(source), dtype=np.bool_)
    weapon = np.zeros(len(source), dtype=np.bool_)
    builder = np.zeros(len(source), dtype=np.bool_)
    present = np.zeros(len(source), dtype=np.bool_)
    spec_mask = np.zeros((len(source), capacity), dtype=np.bool_)
    gather = np.zeros((len(source), capacity), dtype=np.uint8)
    quality = np.zeros((len(source), capacity), dtype=np.int16)
    power = np.zeros((len(source), capacity), dtype=np.float32)
    for row, item in enumerate(source):
        if not isinstance(item, LocalHeldItemTool):
            raise TypeError("held-tool source has wrong type")
        if any(
            not isinstance(value, bool)
            for value in (
                item.valid,
                item.weapon,
                item.builder_tool,
                item.tool_present,
            )
        ):
            raise TypeError("held-tool flags must be boolean")
        if not item.valid:
            continue
        if len(item.specs) > capacity:
            raise ValueError("held item exceeds tool spec capacity")
        if not item.tool_present and item.specs:
            raise ValueError("held item has specs without a tool")
        indices: set[int] = set()
        for slot, spec in enumerate(item.specs):
            if not isinstance(spec, LocalToolSpec):
                raise TypeError("held-tool spec has wrong type")
            if (
                isinstance(spec.gather_type_index, bool)
                or not isinstance(spec.gather_type_index, int)
                or not 0 <= spec.gather_type_index < len(GATHER_TYPES)
                or isinstance(spec.quality, bool)
                or not isinstance(spec.quality, int)
                or not 0 <= spec.quality <= np.iinfo(np.int16).max
                or isinstance(spec.power, bool)
                or not isinstance(spec.power, (int, float))
                or not np.isfinite(spec.power)
                or spec.power <= 0.0
            ):
                raise ValueError("held-tool spec is invalid")
            if spec.gather_type_index in indices:
                raise ValueError("held item has duplicate gather types")
            indices.add(spec.gather_type_index)
            spec_mask[row, slot] = True
            gather[row, slot] = spec.gather_type_index
            quality[row, slot] = spec.quality
            power[row, slot] = spec.power
        available[row] = True
        weapon[row] = item.weapon
        builder[row] = item.builder_tool
        present[row] = item.tool_present
    spec_shape = prefix + (capacity,)
    return BlockToolState(
        available=jnp.asarray(available.reshape(prefix)),
        weapon=jnp.asarray(weapon.reshape(prefix)),
        builder_tool=jnp.asarray(builder.reshape(prefix)),
        tool_present=jnp.asarray(present.reshape(prefix)),
        spec_mask=jnp.asarray(spec_mask.reshape(spec_shape)),
        gather_type_index=jnp.asarray(gather.reshape(spec_shape)),
        quality=jnp.asarray(quality.reshape(spec_shape)),
        power=jnp.asarray(power.reshape(spec_shape)),
    )


def empty_block_tool_state(
    prefix_shape: tuple[int, ...],
    *,
    spec_capacity: int = BLOCK_TOOL_SPEC_CAPACITY,
) -> BlockToolState:
    """Return canonical unavailable tool evidence."""

    capacity = _capacity(
        spec_capacity,
        BLOCK_TOOL_SPEC_CAPACITY,
        "tool spec",
    )
    prefix = _prefix(prefix_shape)
    return BlockToolState(
        available=jnp.zeros(prefix, dtype=jnp.bool_),
        weapon=jnp.zeros(prefix, dtype=jnp.bool_),
        builder_tool=jnp.zeros(prefix, dtype=jnp.bool_),
        tool_present=jnp.zeros(prefix, dtype=jnp.bool_),
        spec_mask=jnp.zeros(prefix + (capacity,), dtype=jnp.bool_),
        gather_type_index=jnp.zeros(prefix + (capacity,), dtype=jnp.uint8),
        quality=jnp.zeros(prefix + (capacity,), dtype=jnp.int16),
        power=jnp.zeros(prefix + (capacity,), dtype=jnp.float32),
    )


def exact_block_mutation_acknowledgement(
    result: MutableBlockUpdateResult,
) -> BlockMutationAcknowledgement:
    """Expose accepted writes only when both geometry values remain exact."""

    if not isinstance(result, MutableBlockUpdateResult):
        raise TypeError("result must be MutableBlockUpdateResult")
    exact = jnp.all(
        ~result.applied
        | (
            result.before.available
            & result.after.available
            & result.before.geometry.exact
            & result.after.geometry.exact
        ),
        axis=1,
    )
    available = result.accepted & exact & ~result.resync_required
    return BlockMutationAcknowledgement(
        available=available,
        accepted=result.accepted,
        applied=result.applied,
        before=result.before,
        after=result.after,
        revision_before=result.revision_before,
        revision_after=result.revision_after,
        stale_base=result.stale_base,
        duplicate_position=result.duplicate_position,
        capacity_exceeded=result.capacity_exceeded,
        invalid=result.invalid | (result.accepted & ~exact),
        resync_required=result.resync_required | (result.accepted & ~exact),
    )


def native_block_interaction_outcome_matches(
    native: NativeBlockInteractionOutcome,
    jax_result: BlockMutationAcknowledgement,
    *,
    health_atol: float = 1.0e-6,
) -> Array:
    """Compare fixture-native mutations with exact JAX acknowledgements."""

    if not isinstance(native, NativeBlockInteractionOutcome):
        raise TypeError("native must be NativeBlockInteractionOutcome")
    if not isinstance(jax_result, BlockMutationAcknowledgement):
        raise TypeError("jax_result must be BlockMutationAcknowledgement")
    shape = jax_result.applied.shape
    if len(shape) != 2:
        raise ValueError("JAX mutation acknowledgement must be [batch, query]")
    for name, value in zip(
        NativeBlockInteractionOutcome._fields,
        native,
        strict=True,
    ):
        if jnp.shape(value) != shape:
            raise ValueError(
                f"native {name} must match the JAX mutation shape"
            )
    if (
        native.available.dtype != jnp.bool_
        or native.placement.dtype != jnp.bool_
        or native.accepted.dtype != jnp.bool_
        or native.mutation_applied.dtype != jnp.bool_
    ):
        raise TypeError("native interaction flags must be boolean")
    if isinstance(health_atol, bool) or health_atol < 0.0:
        raise ValueError("health_atol must be nonnegative")

    native_before_present = native.runtime_block_id_before != 0
    native_after_present = native.runtime_block_id_after != 0
    before = jax_result.before
    after = jax_result.after
    before_identity = (
        before.geometry.block_present == native_before_present
    ) & (
        ~native_before_present
        | (
            before.geometry.runtime_block_id_valid
            & (
                before.geometry.runtime_block_id
                == native.runtime_block_id_before
            )
        )
    )
    after_identity = (
        after.geometry.block_present == native_after_present
    ) & (
        ~native_after_present
        | (
            after.geometry.runtime_block_id_valid
            & (
                after.geometry.runtime_block_id
                == native.runtime_block_id_after
            )
        )
    )
    before_health = ~native_before_present | (
        before.block_health_valid
        & jnp.isclose(
            before.block_health,
            native.block_health_before,
            rtol=0.0,
            atol=health_atol,
        )
    )
    after_health = ~native_after_present | (
        after.block_health_valid
        & jnp.isclose(
            after.block_health,
            native.block_health_after,
            rtol=0.0,
            atol=health_atol,
        )
    )
    native_semantics = jnp.where(
        native.placement,
        ~native_before_present & native_after_present,
        native_before_present
        & (
            ~native_after_present
            | (native.block_health_after < native.block_health_before)
        ),
    )
    available = (
        native.available
        & native.accepted
        & native.mutation_applied
        & jax_result.available[:, None]
        & jax_result.applied
    )
    return (
        available
        & native_semantics
        & before_identity
        & after_identity
        & before_health
        & after_health
    )
