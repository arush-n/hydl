"""Asset-derived fixed-shape item programs for Arsenal abilities.

The host compiler resolves authored ``ModifyInventory`` nodes by semantic
ability ID.  The device runtime then executes the node at the same scheduler
boundary as the surrounding interaction graph.  Nothing in this module
branches on a weapon class or profile name at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.combat.assets.catalog import (
    HYTALE_0_5_7_ASSETS_SHA256,
    HYTALE_0_5_7_COMBAT_CATALOG_SHA256,
)
from hytalegym.jax.combat.arsenal.programs.scheduling import (
    event_dispatch_delay_ticks,
    scheduler_clock,
)
from hytalegym.jax.combat.arsenal.schema.types import AbilityLoadout, ArsenalState
from hytalegym.jax.combat.inventory.schema.contract import (
    CONTAINER_HOTBAR,
    EMPTY_ITEM_ID,
    METADATA_HASH_WORDS,
)
from hytalegym.jax.combat.inventory.runtime.item_interactions import (
    DEFAULT_ITEM_MAX_STACK,
    ModifyInventoryRequest,
    apply_modify_inventory,
)
from hytalegym.jax.combat.inventory.schema.types import InventoryLayout, InventoryState
from hytalegym.jax.combat.contracts.semantic import semantic_id


ABILITY_CHARGE_TIME = "ability_charge_time"
ITEM_PROGRAM_TIME_ORIGIN_ABILITY_SCHEDULER = "ability_scheduler"
ITEM_PROGRAM_TIME_ORIGIN_REQUESTED_CHARGE = "requested_charge"


ITEM_PROGRAM_CATALOG_SCHEMA = "hytalerl_combat_item_program_catalog_v1"
ITEM_PROGRAM_CATALOG_VERSION = 1
ITEM_PROGRAM_STATUS_SUPPORTED = "supported"
ITEM_PROGRAM_STATUS_REQUIRES_UNMODELLED_RELOAD_GRAPH = (
    "requires_unmodelled_reload_graph"
)
NATIVE_ITEM_INTERACTION_EVIDENCE_SCHEMA = (
    "hytalerl_native_item_interaction_evidence_v3"
)
NATIVE_ITEM_INTERACTION_EVIDENCE_VERSION = 3

_CATALOG_PATH = (
    Path(__file__).with_name("assets")
    / "hytale_0_5_7_item_programs_v1.json"
)


@dataclass(frozen=True)
class ItemProgramCatalogRow:
    """One authored ability-to-node binding retained for audit reporting."""

    profile: str
    ability_asset_id: str
    template_id: str
    supported: bool
    status: str
    source_paths: tuple[str, ...]
    graph_position: str


@dataclass(frozen=True)
class ItemProgramCatalog:
    """Validated host-side catalog and its immutable provenance."""

    assets_sha256: str
    combat_asset_catalog_sha256: str
    native_evidence_schema: str
    native_evidence_version: int
    semantic_sha256: str
    program_capacity_per_ability: int
    rows: tuple[ItemProgramCatalogRow, ...]
    document: dict[str, Any]


class AbilityItemPrograms(NamedTuple):
    """Device program bank with axes ``[batch, entity, ability, program]``."""

    program_mask: jax.Array
    program_supported: jax.Array
    time_seconds: jax.Array
    repeat_count: jax.Array
    repeat_interval_seconds: jax.Array
    repeat_continuations_after_ability_events: jax.Array
    failure_delay_seconds: jax.Array
    outer_root_dispatch_scheduler_tick: jax.Array
    dispatch_flags: jax.Array
    required_game_mode: jax.Array
    required_resource_below_max_id: jax.Array
    training_seed_quantity: jax.Array
    remove_item_id: jax.Array
    remove_quantity: jax.Array
    remove_durability: jax.Array
    remove_max_durability: jax.Array
    remove_metadata_hash: jax.Array
    adjust_held_item_quantity: jax.Array
    add_item_id: jax.Array
    add_quantity: jax.Array
    add_durability: jax.Array
    add_max_durability: jax.Array
    add_metadata_hash: jax.Array
    add_item_max_stack: jax.Array
    broken_item_specified: jax.Array
    broken_item_id: jax.Array
    broken_item_max_durability: jax.Array
    adjust_held_item_durability: jax.Array
    failure_suppresses_all_events: jax.Array
    ability_supported: jax.Array
    completion_resource_id: jax.Array
    completion_delay_seconds: jax.Array
    weapon_max_durability: jax.Array
    weapon_durability_loss_on_hit: jax.Array
    weapon_item_id_table: jax.Array
    weapon_item_durability_loss_on_hit_table: jax.Array


class ItemProgramExecution(NamedTuple):
    """State and local graph outcome for one Arsenal microtick."""

    arsenal: ArsenalState
    inventory: InventoryState
    failed: jax.Array
    changed: jax.Array
    dropped_quantity: jax.Array
    suppress_all_events: jax.Array
    failure_delay_seconds: jax.Array


@lru_cache(maxsize=1)
def hytale_0_5_7_item_program_catalog() -> ItemProgramCatalog:
    """Load and fully validate the pinned 0.5.7 item-program evidence."""

    document = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("item-program catalog root must be an object")
    expected_hash = _required_text(document, "semantic_sha256").upper()
    canonical = dict(document)
    canonical.pop("semantic_sha256")
    actual_hash = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest().upper()
    if actual_hash != expected_hash:
        raise ValueError(
            "item-program semantic SHA-256 mismatch: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    _expect(document, "schema", ITEM_PROGRAM_CATALOG_SCHEMA)
    _expect(document, "version", ITEM_PROGRAM_CATALOG_VERSION)
    _expect(document, "assets_sha256", HYTALE_0_5_7_ASSETS_SHA256)
    _expect(
        document,
        "combat_asset_catalog_sha256",
        HYTALE_0_5_7_COMBAT_CATALOG_SHA256,
    )
    _expect(
        document,
        "native_evidence_schema",
        NATIVE_ITEM_INTERACTION_EVIDENCE_SCHEMA,
    )
    _expect(
        document,
        "native_evidence_version",
        NATIVE_ITEM_INTERACTION_EVIDENCE_VERSION,
    )
    capacity = _required_positive_int(document, "program_capacity_per_ability")
    templates = _validated_templates(document)
    bindings = document.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("item-program bindings must be a non-empty list")

    rows: list[ItemProgramCatalogRow] = []
    seen: set[tuple[str, str]] = set()
    ability_templates: dict[str, str] = {}
    for raw in bindings:
        if not isinstance(raw, dict):
            raise ValueError("item-program binding must be an object")
        profile = _required_text(raw, "profile")
        ability = _required_text(raw, "ability_asset_id")
        template_id = _required_text(raw, "template")
        if template_id not in templates:
            raise ValueError(f"unknown item-program template {template_id!r}")
        key = (profile, ability)
        if key in seen:
            raise ValueError(f"duplicate item-program binding {key!r}")
        seen.add(key)
        previous = ability_templates.setdefault(ability, template_id)
        if previous != template_id:
            raise ValueError(
                f"ability {ability!r} maps to conflicting item programs"
            )
        template = templates[template_id]
        status = _required_text(template, "status")
        supported = _required_bool(template, "supported")
        if supported != (status == ITEM_PROGRAM_STATUS_SUPPORTED):
            raise ValueError(
                f"template {template_id!r} has inconsistent status/support"
            )
        source_paths = template.get("source_paths")
        if (
            not isinstance(source_paths, list)
            or not source_paths
            or not all(isinstance(path, str) and path for path in source_paths)
        ):
            raise ValueError(
                f"template {template_id!r} must cite source paths"
            )
        rows.append(
            ItemProgramCatalogRow(
                profile=profile,
                ability_asset_id=ability,
                template_id=template_id,
                supported=supported,
                status=status,
                source_paths=tuple(source_paths),
                graph_position=_required_text(template, "graph_position"),
            )
        )

    _validate_weapon_items(document)
    return ItemProgramCatalog(
        assets_sha256=_required_text(document, "assets_sha256").upper(),
        combat_asset_catalog_sha256=_required_text(
            document,
            "combat_asset_catalog_sha256",
        ).upper(),
        native_evidence_schema=_required_text(
            document,
            "native_evidence_schema",
        ),
        native_evidence_version=int(document["native_evidence_version"]),
        semantic_sha256=expected_hash,
        program_capacity_per_ability=capacity,
        rows=tuple(rows),
        document=document,
    )


def pack_item_programs_for_loadout(
    loadout: AbilityLoadout,
) -> AbilityItemPrograms:
    """Compile catalog rows into the concrete loadout's static device shape."""

    if not isinstance(loadout, AbilityLoadout):
        raise TypeError("loadout must be an AbilityLoadout")
    catalog = hytale_0_5_7_item_program_catalog()
    ability_id = np.asarray(loadout.ability_id, dtype=np.int32)
    ability_mask = np.asarray(loadout.ability_mask, dtype=np.bool_)
    weapon_id = np.asarray(loadout.weapon_id, dtype=np.int32)
    if ability_id.shape != ability_mask.shape or ability_id.ndim != 3:
        raise ValueError("loadout ability ID/mask fields must share shape [B,N,A]")
    if weapon_id.shape != ability_id.shape[:2]:
        raise ValueError("loadout weapon ID field must have shape [B,N]")
    program_shape = ability_id.shape + (catalog.program_capacity_per_ability,)
    metadata_shape = program_shape + (METADATA_HASH_WORDS,)

    program_mask = np.zeros(program_shape, dtype=np.bool_)
    program_supported = np.zeros(program_shape, dtype=np.bool_)
    time_seconds = np.zeros(program_shape, dtype=np.float32)
    repeat_counts = np.ones(program_shape, dtype=np.int32)
    repeat_interval_seconds = np.zeros(program_shape, dtype=np.float32)
    repeat_continuations_after_ability_events = np.zeros(
        program_shape,
        dtype=np.bool_,
    )
    failure_delay_seconds = np.zeros(program_shape, dtype=np.float32)
    outer_root_dispatch_scheduler_tick = np.full(
        program_shape,
        np.iinfo(np.int32).min,
        dtype=np.int32,
    )
    dispatch_flags = np.zeros(program_shape, dtype=np.uint32)
    required_game_mode = np.full(program_shape, -1, dtype=np.int32)
    required_resource_below_max_id = np.full(
        program_shape,
        -1,
        dtype=np.int32,
    )
    training_seed_quantity = np.zeros(program_shape, dtype=np.int32)
    remove_item_id = np.full(program_shape, EMPTY_ITEM_ID, dtype=np.int32)
    remove_quantity = np.zeros(program_shape, dtype=np.int32)
    remove_durability = np.zeros(program_shape, dtype=np.float32)
    remove_max_durability = np.zeros(program_shape, dtype=np.float32)
    remove_metadata_hash = np.zeros(metadata_shape, dtype=np.uint32)
    adjust_held_quantity = np.zeros(program_shape, dtype=np.int32)
    add_item_id = np.full(program_shape, EMPTY_ITEM_ID, dtype=np.int32)
    add_quantity = np.zeros(program_shape, dtype=np.int32)
    add_durability = np.zeros(program_shape, dtype=np.float32)
    add_max_durability = np.zeros(program_shape, dtype=np.float32)
    add_metadata_hash = np.zeros(metadata_shape, dtype=np.uint32)
    add_item_max_stack = np.full(
        program_shape,
        DEFAULT_ITEM_MAX_STACK,
        dtype=np.int32,
    )
    broken_item_specified = np.zeros(program_shape, dtype=np.bool_)
    broken_item_id = np.full(program_shape, EMPTY_ITEM_ID, dtype=np.int32)
    broken_item_max_durability = np.zeros(program_shape, dtype=np.float32)
    adjust_held_durability = np.zeros(program_shape, dtype=np.float32)
    failure_suppresses_all_events = np.zeros(program_shape, dtype=np.bool_)
    # Unsupported catalog rows describe explicitly held graph boundaries;
    # their leaf abilities remain executable unless the profile says otherwise.
    ability_supported = np.ones(ability_id.shape, dtype=np.bool_)
    completion_resource_id = np.full(ability_id.shape, -1, dtype=np.int32)
    completion_delay_seconds = np.zeros(ability_id.shape, dtype=np.float32)

    templates = {
        template["id"]: template
        for template in catalog.document["templates"]
    }
    assigned = np.zeros(ability_id.shape, dtype=np.bool_)
    unique_bindings: dict[str, ItemProgramCatalogRow] = {}
    for row in catalog.rows:
        unique_bindings.setdefault(row.ability_asset_id, row)
    for row in unique_bindings.values():
        template = templates[row.template_id]
        match = ability_mask & (
            ability_id == np.int32(semantic_id(row.ability_asset_id))
        )
        if not np.any(match):
            continue
        if np.any(assigned & match):
            raise ValueError(
                f"loadout ability {row.ability_asset_id!r} received multiple programs"
            )
        assigned |= match
        if not row.supported:
            continue
        template_repeat_count = int(template.get("repeat_count", 1))
        if template_repeat_count <= 0:
            raise ValueError(f"template {row.template_id!r} repeat_count must be positive")
        repeat_interval = np.float32(
            template.get("repeat_interval_seconds", 0.0)
        )
        required_resource = np.int32(
            template.get("required_resource_below_max_id", -1)
        )
        completion_resource_id = np.where(
            match,
            np.int32(template.get("completion_resource_id", -1)),
            completion_resource_id,
        )
        completion_delay_seconds = np.where(
            match,
            np.float32(template.get("completion_delay_seconds", 0.0)),
            completion_delay_seconds,
        )
        raw_time = template.get("time_seconds")
        if raw_time == ABILITY_CHARGE_TIME:
            authored_time = np.asarray(
                loadout.ability_charge_times_seconds[..., 0],
                dtype=np.float32,
            )
        else:
            authored_time = np.float32(
                _required_number(template, "time_seconds")
            )
        time_origin = template.get(
            "time_origin",
            ITEM_PROGRAM_TIME_ORIGIN_ABILITY_SCHEDULER,
        )
        if time_origin == ITEM_PROGRAM_TIME_ORIGIN_REQUESTED_CHARGE:
            requested_charge = np.asarray(
                loadout.ability_requested_charge_time_seconds,
                dtype=np.float32,
            )
            invalid_origin = match & (requested_charge < np.float32(0.0))
            if np.any(invalid_origin):
                raise ValueError(
                    "requested-charge item clock requires an explicit "
                    "ability charge request"
                )
            before_origin = match & (
                authored_time + np.float32(1.0e-6) < requested_charge
            )
            if np.any(before_origin):
                raise ValueError(
                    "item-program time cannot precede its requested-charge "
                    "clock origin"
                )
            authored_time = np.maximum(
                authored_time - requested_charge,
                np.float32(0.0),
            )
        remove_asset = _optional_text(template, "remove_item_asset_id")
        add_asset = _optional_text(template, "add_item_asset_id")
        broken_asset = _optional_text(template, "broken_item_asset_id")
        index = (..., 0)
        program_mask[index] |= match
        program_supported[index] |= match
        time_seconds[index] = np.where(match, authored_time, time_seconds[index])
        repeat_counts[index] = np.where(
            match,
            np.int32(template_repeat_count),
            repeat_counts[index],
        )
        repeat_interval_seconds[index] = np.where(
            match,
            repeat_interval,
            repeat_interval_seconds[index],
        )
        repeat_continuations_after_ability_events[index] = np.where(
            match,
            bool(template.get("repeat_continuations_after_ability_events", False)),
            repeat_continuations_after_ability_events[index],
        )
        failure_delay_seconds[index] = np.where(
            match,
            np.float32(template.get("failure_delay_seconds", 0.0)),
            failure_delay_seconds[index],
        )
        authored_outer_dispatch_tick = np.asarray(
            loadout.ability_outer_root_item_dispatch_tick,
            dtype=np.int32,
        )
        outer_dispatch_tick = authored_outer_dispatch_tick - np.asarray(
            loadout.ability_scheduler_prelude_ticks,
            dtype=np.int32,
        )
        outer_root_dispatch_scheduler_tick[index] = np.where(
            match & np.asarray(
                loadout.ability_outer_root_selector,
                dtype=np.bool_,
            ) & (authored_outer_dispatch_tick >= 0),
            outer_dispatch_tick,
            outer_root_dispatch_scheduler_tick[index],
        )
        dispatch_flags[index] = np.where(
            match,
            np.uint32(_required_nonnegative_int(template, "dispatch_flags")),
            dispatch_flags[index],
        )
        required_game_mode[index] = np.where(
            match,
            np.int32(_required_int(template, "required_game_mode")),
            required_game_mode[index],
        )
        required_resource_below_max_id[index] = np.where(
            match,
            required_resource,
            required_resource_below_max_id[index],
        )
        training_seed_quantity[index] = np.where(
            match,
            np.int32(template.get("training_seed_quantity", 0)),
            training_seed_quantity[index],
        )
        remove_item_id[index] = np.where(
            match,
            np.int32(semantic_id(remove_asset)) if remove_asset else EMPTY_ITEM_ID,
            remove_item_id[index],
        )
        remove_quantity[index] = np.where(
            match,
            np.int32(_required_nonnegative_int(template, "remove_quantity")),
            remove_quantity[index],
        )
        adjust_held_quantity[index] = np.where(
            match,
            np.int32(_required_int(template, "adjust_held_item_quantity")),
            adjust_held_quantity[index],
        )
        add_item_id[index] = np.where(
            match,
            np.int32(semantic_id(add_asset)) if add_asset else EMPTY_ITEM_ID,
            add_item_id[index],
        )
        add_quantity[index] = np.where(
            match,
            np.int32(_required_nonnegative_int(template, "add_quantity")),
            add_quantity[index],
        )
        broken_item_specified[index] = np.where(
            match,
            bool(broken_asset),
            broken_item_specified[index],
        )
        broken_item_id[index] = np.where(
            match,
            np.int32(semantic_id(broken_asset)) if broken_asset else EMPTY_ITEM_ID,
            broken_item_id[index],
        )
        adjust_held_durability[index] = np.where(
            match,
            np.float32(_required_number(template, "adjust_held_item_durability")),
            adjust_held_durability[index],
        )
        failure_suppresses_all_events[index] = np.where(
            match,
            _required_bool(template, "failure_suppresses_all_ability_events"),
            failure_suppresses_all_events[index],
        )

    weapon_maximum = np.zeros(weapon_id.shape, dtype=np.float32)
    weapon_durability_loss = np.zeros(weapon_id.shape, dtype=np.float32)
    weapon_items = catalog.document["weapon_items"]
    weapon_item_id_table = np.asarray(
        [
            semantic_id(_required_text(item, "item_asset_id"))
            for item in weapon_items
        ],
        dtype=np.int32,
    )
    if np.unique(weapon_item_id_table).size != weapon_item_id_table.size:
        raise ValueError("weapon item semantic IDs must be collision-free")
    weapon_item_durability_loss_table = np.asarray(
        [
            _required_number(item, "durability_loss_on_hit")
            for item in weapon_items
        ],
        dtype=np.float32,
    )
    for item, item_id in zip(
        weapon_items,
        weapon_item_id_table,
        strict=True,
    ):
        selected = weapon_id == item_id
        weapon_maximum = np.where(
            selected,
            np.float32(_required_number(item, "max_durability")),
            weapon_maximum,
        )
        weapon_durability_loss = np.where(
            selected,
            np.float32(_required_number(item, "durability_loss_on_hit")),
            weapon_durability_loss,
        )

    return AbilityItemPrograms(
        program_mask=jnp.asarray(program_mask),
        program_supported=jnp.asarray(program_supported),
        time_seconds=jnp.asarray(time_seconds),
        repeat_count=jnp.asarray(repeat_counts),
        repeat_interval_seconds=jnp.asarray(repeat_interval_seconds),
        repeat_continuations_after_ability_events=jnp.asarray(
            repeat_continuations_after_ability_events
        ),
        failure_delay_seconds=jnp.asarray(failure_delay_seconds),
        outer_root_dispatch_scheduler_tick=jnp.asarray(
            outer_root_dispatch_scheduler_tick
        ),
        dispatch_flags=jnp.asarray(dispatch_flags),
        required_game_mode=jnp.asarray(required_game_mode),
        required_resource_below_max_id=jnp.asarray(
            required_resource_below_max_id
        ),
        training_seed_quantity=jnp.asarray(training_seed_quantity),
        remove_item_id=jnp.asarray(remove_item_id),
        remove_quantity=jnp.asarray(remove_quantity),
        remove_durability=jnp.asarray(remove_durability),
        remove_max_durability=jnp.asarray(remove_max_durability),
        remove_metadata_hash=jnp.asarray(remove_metadata_hash),
        adjust_held_item_quantity=jnp.asarray(adjust_held_quantity),
        add_item_id=jnp.asarray(add_item_id),
        add_quantity=jnp.asarray(add_quantity),
        add_durability=jnp.asarray(add_durability),
        add_max_durability=jnp.asarray(add_max_durability),
        add_metadata_hash=jnp.asarray(add_metadata_hash),
        add_item_max_stack=jnp.asarray(add_item_max_stack),
        broken_item_specified=jnp.asarray(broken_item_specified),
        broken_item_id=jnp.asarray(broken_item_id),
        broken_item_max_durability=jnp.asarray(
            broken_item_max_durability
        ),
        adjust_held_item_durability=jnp.asarray(adjust_held_durability),
        failure_suppresses_all_events=jnp.asarray(
            failure_suppresses_all_events
        ),
        ability_supported=jnp.asarray(ability_supported),
        completion_resource_id=jnp.asarray(completion_resource_id),
        completion_delay_seconds=jnp.asarray(completion_delay_seconds),
        weapon_max_durability=jnp.asarray(weapon_maximum),
        weapon_durability_loss_on_hit=jnp.asarray(weapon_durability_loss),
        weapon_item_id_table=jnp.asarray(weapon_item_id_table),
        weapon_item_durability_loss_on_hit_table=jnp.asarray(
            weapon_item_durability_loss_table
        ),
    )


def execute_due_item_programs(
    arsenal: ArsenalState,
    inventory: InventoryState,
    layout: InventoryLayout,
    programs: AbilityItemPrograms,
    after_scheduler_clocks: jax.Array,
    active_game_mode: jax.Array = jnp.int32(0),
    resources: jax.Array | None = None,
    resource_maximum: jax.Array | None = None,
    continuation_phase: bool = False,
    player_backed_actor_mask: jax.Array | bool = False,
) -> ItemProgramExecution:
    """Execute due nodes and follow a local Failed edge by cancelling a child.

    A missing required stack fails only its interaction branch.  It does not
    set Arsenal or inventory failure bits and therefore does not truncate the
    environment. Earlier fields/nodes remain committed, and the returned
    suppression signal names whether this branch owns all modeled events.

    ``after_scheduler_clocks`` must be the single prior-to-after boundary used
    by the surrounding ability tick.  This function never advances clocks,
    which prevents an integration call after ``tick_ability_programs`` from
    silently applying a second scheduler step.
    """

    batch, entity_count = arsenal.active_ability_slot.shape
    if inventory.active_hotbar_slot.shape != (batch, entity_count):
        raise ValueError("inventory and Arsenal batch/entity axes must match")
    ability_capacity = programs.program_mask.shape[2]
    if programs.program_mask.shape[:2] != (batch, entity_count):
        raise ValueError("item programs and Arsenal batch/entity axes must match")
    if (resources is None) != (resource_maximum is None):
        raise ValueError("resources and resource_maximum must be supplied together")
    if not isinstance(continuation_phase, bool):
        raise TypeError("continuation_phase must be a bool")
    resource_values = None
    resource_limits = None
    if resources is not None:
        resource_values = jnp.asarray(resources, dtype=jnp.float32)
        resource_limits = jnp.asarray(resource_maximum, dtype=jnp.float32)
        if resource_values.ndim != 3 or resource_values.shape[:2] != (
            batch,
            entity_count,
        ):
            raise ValueError("resources must have shape [batch, entity, resource]")
        if resource_limits.shape != resource_values.shape:
            raise ValueError("resource_maximum must match resources")
    game_mode = _broadcast(
        active_game_mode,
        (batch, entity_count),
        jnp.int32,
        "active_game_mode",
    )
    player_backed = _broadcast(
        player_backed_actor_mask,
        (batch, entity_count),
        jnp.bool_,
        "player_backed_actor_mask",
    )
    active = arsenal.active_ability_slot >= 0
    slot = jnp.clip(
        arsenal.active_ability_slot,
        jnp.int32(0),
        jnp.int32(ability_capacity - 1),
    )
    after_clocks = jnp.asarray(
        after_scheduler_clocks,
        dtype=jnp.float32,
    )
    if after_clocks.shape != arsenal.ability_scheduler_clock_seconds.shape:
        raise ValueError(
            "after_scheduler_clocks must match Arsenal scheduler clocks"
        )

    mask = _gather_ability(programs.program_mask, slot)
    supported = _gather_ability(programs.program_supported, slot)
    time_seconds = _gather_ability(programs.time_seconds, slot)
    repeats = _gather_ability(programs.repeat_count, slot)
    repeat_interval = _gather_ability(
        programs.repeat_interval_seconds,
        slot,
    )
    continuation_after_events = _gather_ability(
        programs.repeat_continuations_after_ability_events,
        slot,
    )
    flags = _gather_ability(programs.dispatch_flags, slot)
    outer_dispatch_tick = _gather_ability(
        programs.outer_root_dispatch_scheduler_tick,
        slot,
    )
    delay = event_dispatch_delay_ticks(
        time_seconds,
        flags,
        player_backed,
    )
    prior = scheduler_clock(arsenal.ability_scheduler_clock_seconds, delay)
    after = scheduler_clock(after_clocks, delay)
    first_crossed = (
        ((time_seconds == jnp.float32(0.0)) & (prior == jnp.float32(0.0)))
        | ((time_seconds > prior) & (time_seconds <= after))
    )
    outer_dispatch_enabled = outer_dispatch_tick != jnp.iinfo(jnp.int32).min
    outer_dispatch_crossed = (
        (arsenal.ability_scheduler_tick[..., None] < outer_dispatch_tick)
        & (
            arsenal.ability_scheduler_tick[..., None] + jnp.int32(1)
            >= outer_dispatch_tick
        )
    )
    first_crossed = jnp.where(
        outer_dispatch_enabled,
        outer_dispatch_crossed,
        first_crossed,
    )
    safe_interval = jnp.maximum(repeat_interval, jnp.float32(1.0e-6))

    def continuation_count(clock):
        reached = jnp.floor(
            (clock - time_seconds + jnp.float32(1.0e-6)) / safe_interval
        ).astype(jnp.int32)
        return jnp.clip(reached, jnp.int32(0), jnp.maximum(repeats - 1, 0))

    crossed_continuations = jnp.maximum(
        continuation_count(after) - continuation_count(prior),
        jnp.int32(0),
    )
    if continuation_phase:
        execution_count = jnp.where(
            continuation_after_events & (repeats > 1),
            crossed_continuations,
            jnp.int32(0),
        )
    else:
        execution_count = first_crossed.astype(jnp.int32)
    # An outer-root inventory node may execute while the selected child's
    # scheduler clocks are still held behind its prelude. Its integer root
    # boundary is therefore sufficient; ordinary child programs still
    # require a real scheduler-clock advance.
    dispatch_advanced = jnp.where(
        outer_dispatch_enabled,
        jnp.bool_(True),
        after > prior,
    )
    due = active[..., None] & mask & dispatch_advanced & (execution_count > 0)

    current = inventory
    failed = jnp.zeros((batch, entity_count), dtype=jnp.bool_)
    changed = jnp.zeros_like(failed)
    suppress_all_events = jnp.zeros_like(failed)
    failure_delay_seconds = jnp.zeros_like(failed, dtype=jnp.float32)
    dropped = jnp.zeros((batch, entity_count), dtype=jnp.int32)
    held_container = jnp.full(
        (batch, entity_count),
        CONTAINER_HOTBAR,
        dtype=jnp.int32,
    )
    program_capacity = programs.program_mask.shape[3]
    for program_index in range(program_capacity):
        request_mask = due[..., program_index] & ~failed
        required_resource = _gather_program(
            programs.required_resource_below_max_id,
            slot,
            program_index,
        )
        if resource_values is not None:
            safe_resource = jnp.clip(
                required_resource,
                jnp.int32(0),
                jnp.int32(resource_values.shape[2] - 1),
            )
            current_resource = jnp.take_along_axis(
                resource_values,
                safe_resource[..., None],
                axis=2,
            )[..., 0]
            maximum_resource = jnp.take_along_axis(
                resource_limits,
                safe_resource[..., None],
                axis=2,
            )[..., 0]
            request_mask &= (required_resource < 0) | (
                current_resource + jnp.float32(1.0e-6) < maximum_resource
            )
        unsupported = request_mask & ~supported[..., program_index]
        failed |= unsupported
        request_mask &= supported[..., program_index]
        count = execution_count[..., program_index]
        request = ModifyInventoryRequest(
            required_game_mode=_gather_program(
                programs.required_game_mode,
                slot,
                program_index,
            ),
            remove_item_id=_gather_program(
                programs.remove_item_id,
                slot,
                program_index,
            ),
            remove_quantity=_gather_program(
                programs.remove_quantity,
                slot,
                program_index,
            ) * count,
            remove_durability=_gather_program(
                programs.remove_durability,
                slot,
                program_index,
            ),
            remove_max_durability=_gather_program(
                programs.remove_max_durability,
                slot,
                program_index,
            ),
            remove_metadata_hash=_gather_program(
                programs.remove_metadata_hash,
                slot,
                program_index,
            ),
            adjust_held_item_quantity=_gather_program(
                programs.adjust_held_item_quantity,
                slot,
                program_index,
            ) * count,
            add_item_id=_gather_program(
                programs.add_item_id,
                slot,
                program_index,
            ),
            add_quantity=_gather_program(
                programs.add_quantity,
                slot,
                program_index,
            ) * count,
            add_durability=_gather_program(
                programs.add_durability,
                slot,
                program_index,
            ),
            add_max_durability=_gather_program(
                programs.add_max_durability,
                slot,
                program_index,
            ),
            add_metadata_hash=_gather_program(
                programs.add_metadata_hash,
                slot,
                program_index,
            ),
            add_item_max_stack=_gather_program(
                programs.add_item_max_stack,
                slot,
                program_index,
            ),
            broken_item_specified=_gather_program(
                programs.broken_item_specified,
                slot,
                program_index,
            ),
            broken_item_id=_gather_program(
                programs.broken_item_id,
                slot,
                program_index,
            ),
            broken_item_max_durability=_gather_program(
                programs.broken_item_max_durability,
                slot,
                program_index,
            ),
            adjust_held_item_durability=_gather_program(
                programs.adjust_held_item_durability,
                slot,
                program_index,
            ) * count.astype(jnp.float32),
        )
        result = apply_modify_inventory(
            current,
            layout,
            request,
            request_mask=request_mask,
            active_game_mode=game_mode,
            held_container_id=held_container,
            held_container_slot=current.active_hotbar_slot,
        )
        current = result.state
        program_failed = unsupported | result.failed
        failure_delay_seconds = jnp.maximum(
            failure_delay_seconds,
            jnp.where(
                program_failed,
                _gather_program(
                    programs.failure_delay_seconds,
                    slot,
                    program_index,
                ),
                jnp.float32(0.0),
            ),
        )
        failed |= result.failed
        suppress_all_events |= result.failed & _gather_program(
            programs.failure_suppresses_all_events,
            slot,
            program_index,
        )
        changed |= result.changed
        dropped += result.dropped_quantity
    return ItemProgramExecution(
        arsenal=arsenal,
        inventory=current,
        failed=failed,
        changed=changed,
        dropped_quantity=dropped,
        suppress_all_events=suppress_all_events,
        failure_delay_seconds=failure_delay_seconds,
    )


def defer_item_program_completion(
    arsenal: ArsenalState,
    delayed: jax.Array,
    delay_seconds: jax.Array,
    ability_duration_seconds: jax.Array,
    dt_seconds: jax.Array,
    *,
    advance_occurs_this_tick: bool = False,
) -> ArsenalState:
    """Advance graph clocks so an authored failure child runs to completion.

    This retains learner-visible elapsed time while skipping Repeat work that
    can no longer execute. Duration boundaries round up to whole ticks; rates
    are not rounded.
    """

    batch, entity_count = arsenal.active_ability_slot.shape
    mask = jnp.asarray(delayed, dtype=jnp.bool_)
    delay = jnp.asarray(delay_seconds, dtype=jnp.float32)
    if mask.shape != (batch, entity_count) or delay.shape != mask.shape:
        raise ValueError("delayed and delay_seconds must have shape [B,N]")
    duration_bank = jnp.asarray(ability_duration_seconds, dtype=jnp.float32)
    if duration_bank.shape[:2] != (batch, entity_count) or duration_bank.ndim != 3:
        raise ValueError("ability_duration_seconds must have shape [B,N,A]")
    dt = jnp.asarray(dt_seconds, dtype=jnp.float32)
    if dt.ndim == 0:
        dt = jnp.broadcast_to(dt, mask.shape)
    elif dt.shape == (batch,):
        dt = jnp.broadcast_to(dt[:, None], mask.shape)
    if dt.shape != mask.shape:
        raise ValueError("dt_seconds must be scalar, [B], or [B,N]")
    ability_capacity = duration_bank.shape[2]
    slot = jnp.clip(
        arsenal.active_ability_slot,
        jnp.int32(0),
        jnp.int32(ability_capacity - 1),
    )
    duration = _gather_ability(duration_bank, slot)
    safe_dt = jnp.maximum(dt, jnp.float32(1.0e-9))
    if not isinstance(advance_occurs_this_tick, bool):
        raise TypeError("advance_occurs_this_tick must be a bool")
    delay_ticks = jnp.ceil(delay / safe_dt).astype(jnp.int32)
    remaining_advances = jnp.maximum(
        delay_ticks
        - jnp.where(
            advance_occurs_this_tick,
            jnp.int32(0),
            jnp.int32(1),
        ),
        jnp.int32(0),
    )
    target = jnp.maximum(
        jnp.float32(0.0),
        duration
        - remaining_advances.astype(jnp.float32) * safe_dt,
    )
    clocks = jnp.where(
        mask[..., None],
        jnp.maximum(
            arsenal.ability_scheduler_clock_seconds,
            target[..., None],
        ),
        arsenal.ability_scheduler_clock_seconds,
    )
    return arsenal._replace(ability_scheduler_clock_seconds=clocks)


def complete_resource_bound_item_programs(
    arsenal: ArsenalState,
    programs: AbilityItemPrograms,
    prior_resources: jax.Array,
    resources: jax.Array,
    resource_maximum: jax.Array,
    ability_duration_seconds: jax.Array,
    dt_seconds: jax.Array,
) -> ArsenalState:
    """Follow a Repeat root's authored failed tail after its stat fills."""

    batch, entity_count = arsenal.active_ability_slot.shape
    if programs.completion_resource_id.shape[:2] != (batch, entity_count):
        raise ValueError("item programs and Arsenal batch/entity axes must match")
    prior_values = jnp.asarray(prior_resources, dtype=jnp.float32)
    values = jnp.asarray(resources, dtype=jnp.float32)
    maximum = jnp.asarray(resource_maximum, dtype=jnp.float32)
    if (
        prior_values.shape != values.shape
        or values.shape != maximum.shape
        or values.shape[:2] != (batch, entity_count)
    ):
        raise ValueError(
            "prior_resources, resources, and resource_maximum must share [B,N,R]"
        )
    ability_capacity = programs.completion_resource_id.shape[2]
    slot = jnp.clip(
        arsenal.active_ability_slot,
        jnp.int32(0),
        jnp.int32(ability_capacity - 1),
    )
    resource_id = _gather_ability(programs.completion_resource_id, slot)
    safe_resource = jnp.clip(
        resource_id,
        jnp.int32(0),
        jnp.int32(values.shape[2] - 1),
    )
    current = jnp.take_along_axis(values, safe_resource[..., None], axis=2)[..., 0]
    prior = jnp.take_along_axis(
        prior_values,
        safe_resource[..., None],
        axis=2,
    )[..., 0]
    limit = jnp.take_along_axis(maximum, safe_resource[..., None], axis=2)[..., 0]
    reached = (
        (arsenal.active_ability_slot >= 0)
        & (resource_id >= 0)
        & (prior + jnp.float32(1.0e-6) < limit)
        & (current + jnp.float32(1.0e-6) >= limit)
    )
    delay = _gather_ability(programs.completion_delay_seconds, slot)
    delayed = reached & (delay > jnp.float32(0.0))
    immediate = reached & ~delayed
    deferred = defer_item_program_completion(
        arsenal,
        delayed,
        delay,
        ability_duration_seconds,
        dt_seconds,
    )
    return deferred._replace(
        active_ability_slot=jnp.where(
            immediate,
            jnp.int32(-1),
            deferred.active_ability_slot,
        ),
        active_ability_root_slot=jnp.where(
            immediate,
            jnp.int32(-1),
            deferred.active_ability_root_slot,
        ),
        ability_elapsed_seconds=jnp.where(
            immediate,
            jnp.float32(0.0),
            deferred.ability_elapsed_seconds,
        ),
        ability_scheduler_tick=jnp.where(
            immediate,
            jnp.int32(0),
            deferred.ability_scheduler_tick,
        ),
        ability_scheduler_clock_seconds=jnp.where(
            immediate[..., None],
            jnp.float32(0.0),
            deferred.ability_scheduler_clock_seconds,
        ),
        ability_selector_hit_bits=jnp.where(
            immediate,
            jnp.uint32(0),
            deferred.ability_selector_hit_bits,
        ),
    )


def _gather_ability(value: jax.Array, slot: jax.Array) -> jax.Array:
    trailing = value.shape[3:]
    index = slot.reshape(slot.shape + (1,) * (1 + len(trailing)))
    index = jnp.broadcast_to(index, slot.shape + (1,) + trailing)
    return jnp.take_along_axis(value, index, axis=2)[:, :, 0, ...]


def _gather_program(
    value: jax.Array,
    slot: jax.Array,
    program_index: int,
) -> jax.Array:
    return _gather_ability(value, slot)[:, :, program_index, ...]


def _broadcast(value, shape, dtype, name):
    result = jnp.asarray(value, dtype=dtype)
    if result.ndim == 0:
        result = jnp.broadcast_to(result, shape)
    elif result.shape == (shape[0],) and len(shape) == 2:
        result = jnp.broadcast_to(result[:, None], shape)
    if result.shape != shape:
        raise ValueError(f"{name} must be scalar or have shape {shape}")
    return result


def _validated_templates(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_templates = document.get("templates")
    if not isinstance(raw_templates, list) or not raw_templates:
        raise ValueError("item-program templates must be a non-empty list")
    result: dict[str, dict[str, Any]] = {}
    for template in raw_templates:
        if not isinstance(template, dict):
            raise ValueError("item-program template must be an object")
        template_id = _required_text(template, "id")
        if template_id in result:
            raise ValueError(f"duplicate item-program template {template_id!r}")
        _required_bool(template, "supported")
        _required_bool(
            template,
            "failure_suppresses_all_ability_events",
        )
        _required_text(template, "status")
        time_value = template.get("time_seconds")
        if time_value != ABILITY_CHARGE_TIME:
            _required_number(template, "time_seconds")
        time_origin = template.get(
            "time_origin",
            ITEM_PROGRAM_TIME_ORIGIN_ABILITY_SCHEDULER,
        )
        if time_origin not in {
            ITEM_PROGRAM_TIME_ORIGIN_ABILITY_SCHEDULER,
            ITEM_PROGRAM_TIME_ORIGIN_REQUESTED_CHARGE,
        }:
            raise ValueError(
                "time_origin must be 'ability_scheduler' or "
                "'requested_charge'"
            )
        if (
            time_origin == ITEM_PROGRAM_TIME_ORIGIN_REQUESTED_CHARGE
            and time_value == ABILITY_CHARGE_TIME
        ):
            raise ValueError(
                "ability_charge_time cannot also use requested_charge as "
                "its item-program clock origin"
            )
        _required_nonnegative_int(template, "dispatch_flags")
        _required_int(template, "required_game_mode")
        _required_nonnegative_int(template, "remove_quantity")
        _required_int(template, "adjust_held_item_quantity")
        _required_nonnegative_int(template, "add_quantity")
        _required_number(template, "adjust_held_item_durability")
        repeat_count = template.get("repeat_count", 1)
        if isinstance(repeat_count, bool) or not isinstance(repeat_count, int):
            raise ValueError("repeat_count must be an integer")
        if repeat_count <= 0:
            raise ValueError("repeat_count must be positive")
        repeat_interval = template.get("repeat_interval_seconds", 0.0)
        if isinstance(repeat_interval, bool) or not isinstance(
            repeat_interval,
            (int, float),
        ) or not np.isfinite(repeat_interval) or repeat_interval < 0.0:
            raise ValueError("repeat_interval_seconds must be finite and non-negative")
        if repeat_count > 1 and repeat_interval <= 0.0:
            raise ValueError(
                "repeat_interval_seconds must be positive when repeat_count > 1"
            )
        for optional_delay in (
            "authored_repeat_interval_seconds",
            "failure_delay_seconds",
            "completion_delay_seconds",
        ):
            if optional_delay not in template:
                continue
            value = _required_number(template, optional_delay)
            if value < 0.0:
                raise ValueError(f"{optional_delay} must be non-negative")
        continuation_after = template.get(
            "repeat_continuations_after_ability_events",
            False,
        )
        if not isinstance(continuation_after, bool):
            raise ValueError(
                "repeat_continuations_after_ability_events must be a Boolean"
            )
        if continuation_after and repeat_count <= 1:
            raise ValueError(
                "post-event repeat continuations require repeat_count > 1"
            )
        for optional_id in (
            "required_resource_below_max_id",
            "completion_resource_id",
            "training_seed_quantity",
        ):
            value = template.get(optional_id, -1 if optional_id != "training_seed_quantity" else 0)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{optional_id} must be an integer")
            minimum = 0 if optional_id == "training_seed_quantity" else -1
            if value < minimum:
                raise ValueError(f"{optional_id} must be >= {minimum}")
        result[template_id] = template
    return result


def _validate_weapon_items(document: dict[str, Any]) -> None:
    items = document.get("weapon_items")
    if not isinstance(items, list):
        raise ValueError("weapon_items must be a list")
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("weapon item must be an object")
        item_id = _required_text(item, "item_asset_id")
        if item_id in seen:
            raise ValueError(f"duplicate weapon item {item_id!r}")
        seen.add(item_id)
        if _required_number(item, "max_durability") <= 0.0:
            raise ValueError("weapon max durability must be positive")
        durability_loss = _required_number(item, "durability_loss_on_hit")
        if durability_loss < 0.0:
            raise ValueError("weapon durability loss on hit must be non-negative")
        _required_text(item, "source_path")


def _expect(document: dict[str, Any], key: str, expected: Any) -> None:
    actual = document.get(key)
    if isinstance(expected, str) and isinstance(actual, str):
        equal = actual.upper() == expected.upper() if key.endswith("sha256") else actual == expected
    else:
        equal = actual == expected
    if not equal:
        raise ValueError(f"item-program {key} must be {expected!r}, got {actual!r}")


def _required_text(document: dict[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_text(document: dict[str, Any], key: str) -> str:
    value = document.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _required_bool(document: dict[str, Any], key: str) -> bool:
    value = document.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a Boolean")
    return value


def _required_number(document: dict[str, Any], key: str) -> float:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    if not np.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return float(value)


def _required_int(document: dict[str, Any], key: str) -> int:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _required_nonnegative_int(document: dict[str, Any], key: str) -> int:
    value = _required_int(document, key)
    if value < 0:
        raise ValueError(f"{key} must be non-negative")
    return value


def _required_positive_int(document: dict[str, Any], key: str) -> int:
    value = _required_int(document, key)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


__all__ = [
    "AbilityItemPrograms",
    "ITEM_PROGRAM_CATALOG_SCHEMA",
    "ITEM_PROGRAM_CATALOG_VERSION",
    "ITEM_PROGRAM_STATUS_REQUIRES_UNMODELLED_RELOAD_GRAPH",
    "ITEM_PROGRAM_STATUS_SUPPORTED",
    "ItemProgramCatalog",
    "ItemProgramCatalogRow",
    "ItemProgramExecution",
    "complete_resource_bound_item_programs",
    "defer_item_program_completion",
    "execute_due_item_programs",
    "hytale_0_5_7_item_program_catalog",
    "pack_item_programs_for_loadout",
]
