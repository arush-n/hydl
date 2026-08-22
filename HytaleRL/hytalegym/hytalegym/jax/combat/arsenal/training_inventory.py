"""Scenario-neutral inventory implied by authored Arsenal loadouts."""

from __future__ import annotations

import numpy as np

from hytalegym.jax.combat.arsenal.schema.types import ArsenalRuntimeConfig
from hytalegym.jax.combat.inventory import (
    CONTAINER_HOTBAR,
    CONTAINER_STORAGE,
    EMPTY_ITEM_ID,
    InventoryState,
    inventory_from_loadout,
)
from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.types import AGENT_ENTITY
from hytalegym.jax.combat.inventory.runtime.item_interactions import (
    initialize_held_item_durability,
)


def training_inventory_from_config(
    config: ArsenalRuntimeConfig,
) -> InventoryState:
    """Build the default training inventory without inventing free resources.

    A few native item graphs consume a physical stack in addition to changing
    the corresponding entity stat.  When a profile authors a positive initial
    stat budget, seed the matching external item-program input with the number
    of executions that budget can fund.  Programs without a positive authored
    initial budget stay empty, and held-item removals use the equipped stack
    already created by :func:`inventory_from_loadout`.
    """

    if not isinstance(config, ArsenalRuntimeConfig):
        raise TypeError("config must be an ArsenalRuntimeConfig")
    layout = config.inventory_layout
    inventory = inventory_from_loadout(config.loadout, layout)
    inventory = initialize_held_item_durability(
        inventory,
        layout,
        config.item_programs.weapon_max_durability,
    )

    program_mask = np.asarray(
        config.item_programs.program_mask
        & config.item_programs.program_supported,
        dtype=np.bool_,
    )
    remove_item = np.asarray(config.item_programs.remove_item_id, dtype=np.int32)
    remove_quantity = np.asarray(
        config.item_programs.remove_quantity,
        dtype=np.int32,
    )
    training_seed_quantity = np.asarray(
        config.item_programs.training_seed_quantity,
        dtype=np.int32,
    )
    weapon_id = np.asarray(config.loadout.weapon_id, dtype=np.int32)
    resource_initial = np.asarray(
        config.loadout.resource_initial,
        dtype=np.float32,
    )
    resource_cost = np.asarray(
        config.loadout.ability_resource_cost,
        dtype=np.float32,
    )
    storage_slots = np.flatnonzero(
        np.asarray(layout.container_id) == CONTAINER_STORAGE
    )
    hotbar_slots = np.flatnonzero(
        (np.asarray(layout.container_id) == CONTAINER_HOTBAR)
        & (np.asarray(layout.container_slot) != 0)
    )
    outer_root_selector = np.asarray(
        config.loadout.ability_outer_root_selector,
        dtype=np.bool_,
    )

    item_id = inventory.item_id
    quantity = inventory.quantity
    batch, entities = weapon_id.shape
    for environment in range(batch):
        for entity in range(entities):
            budgets: dict[int, int] = {}
            outer_root_budgets: dict[int, int] = {}
            abilities, programs = program_mask.shape[2:]
            for ability in range(abilities):
                for program in range(programs):
                    if not program_mask[environment, entity, ability, program]:
                        continue
                    asset = int(remove_item[environment, entity, ability, program])
                    seed = int(
                        training_seed_quantity[
                            environment,
                            entity,
                            ability,
                            program,
                        ]
                    )
                    if (
                        asset != EMPTY_ITEM_ID
                        and asset != int(weapon_id[environment, entity])
                        and seed > 0
                    ):
                        budgets[asset] = max(budgets.get(asset, 0), seed)
                    removed = int(
                        remove_quantity[environment, entity, ability, program]
                    )
                    if (
                        outer_root_selector[environment, entity, ability]
                        and asset != EMPTY_ITEM_ID
                        and asset != int(weapon_id[environment, entity])
                        and removed > 0
                    ):
                        # A public outer root owns this prerequisite before
                        # its selected child starts. Seed one exact execution
                        # even when the root-created stat begins at zero.
                        outer_root_budgets[asset] = max(
                            outer_root_budgets.get(asset, 0),
                            removed,
                        )
                costs = resource_cost[environment, entity, ability]
                funded = costs > 0.0
                if not np.any(funded):
                    continue
                executions = int(
                    np.min(
                        np.floor(
                            resource_initial[environment, entity, funded]
                            / costs[funded]
                        )
                    )
                )
                if executions <= 0:
                    continue
                for program in range(programs):
                    if not program_mask[environment, entity, ability, program]:
                        continue
                    asset = int(remove_item[environment, entity, ability, program])
                    removed = int(
                        remove_quantity[environment, entity, ability, program]
                    )
                    if (
                        asset == EMPTY_ITEM_ID
                        or asset == int(weapon_id[environment, entity])
                        or removed <= 0
                    ):
                        continue
                    budgets[asset] = max(
                        budgets.get(asset, 0),
                        executions * removed,
                    )
            for asset in outer_root_budgets:
                budgets.pop(asset, None)
            if len(outer_root_budgets) > len(hotbar_slots):
                raise ValueError(
                    "authored outer-root consumables exceed spare hotbar capacity"
                )
            if len(budgets) > len(storage_slots):
                raise ValueError(
                    "authored initial consumables exceed storage capacity"
                )
            for slot, (asset, count) in zip(
                hotbar_slots,
                sorted(outer_root_budgets.items()),
                strict=False,
            ):
                item_id = item_id.at[environment, entity, int(slot)].set(asset)
                quantity = quantity.at[environment, entity, int(slot)].set(count)
            for slot, (asset, count) in zip(
                storage_slots,
                sorted(budgets.items()),
                strict=False,
            ):
                item_id = item_id.at[environment, entity, int(slot)].set(asset)
                quantity = quantity.at[environment, entity, int(slot)].set(count)

    return inventory._replace(item_id=item_id, quantity=quantity)


def native_training_hotbar_items_from_config(
    config: ArsenalRuntimeConfig,
    *,
    batch_index: int = 0,
    entity_index: int = AGENT_ENTITY,
) -> tuple[tuple[int, str, int], ...]:
    """Return an exact native hotbar for one scenario-neutral actor.

    Native reset options currently expose a hotbar, while JAX keeps six
    distinct containers. Compact the actor's authored starting stacks into
    that bounded transport without branching on a weapon name. Semantic IDs
    are reversed only through the same pinned item-program catalog that
    produced the inventory requirements.
    """

    if not isinstance(config, ArsenalRuntimeConfig):
        raise TypeError("config must be an ArsenalRuntimeConfig")
    if (
        isinstance(batch_index, bool)
        or not isinstance(batch_index, int)
        or not 0 <= batch_index < config.loadout.weapon_id.shape[0]
    ):
        raise ValueError("batch_index is out of range")
    if (
        isinstance(entity_index, bool)
        or not isinstance(entity_index, int)
        or not 0 <= entity_index < config.loadout.weapon_id.shape[1]
    ):
        raise ValueError("entity_index is out of range")

    # Local import keeps catalog construction out of inventory module import
    # order while retaining one authoritative asset-name source.
    from hytalegym.jax.combat.arsenal.item_programs import (
        hytale_0_5_7_item_program_catalog,
    )
    from hytalegym.combat.assets.catalog import COMPILED_PROFILE_SOURCE_ASSETS

    catalog = hytale_0_5_7_item_program_catalog()
    asset_by_semantic_id: dict[int, str] = {}

    def register(asset: object) -> None:
        if not isinstance(asset, str) or not asset:
            return
        value = int(semantic_id(asset))
        previous = asset_by_semantic_id.setdefault(value, asset)
        if previous != asset:
            raise ValueError(
                "item-program semantic ID collision: "
                f"{previous!r} and {asset!r}"
            )

    for item in catalog.document["weapon_items"]:
        register(item.get("item_asset_id"))
    for assets in COMPILED_PROFILE_SOURCE_ASSETS.values():
        for asset in assets:
            register(asset)
    for template in catalog.document["templates"]:
        for key in (
            "remove_item_asset_id",
            "add_item_asset_id",
            "broken_item_asset_id",
        ):
            register(template.get(key))

    inventory = training_inventory_from_config(config)
    item_ids = np.asarray(
        inventory.item_id[batch_index, entity_index],
        dtype=np.int32,
    )
    quantities = np.asarray(
        inventory.quantity[batch_index, entity_index],
        dtype=np.int32,
    )
    stacks: dict[int, int] = {}
    for item, quantity in zip(item_ids, quantities, strict=True):
        value = int(item)
        count = int(quantity)
        if value == EMPTY_ITEM_ID or count <= 0:
            continue
        stacks[value] = stacks.get(value, 0) + count

    equipped = int(config.loadout.weapon_id[batch_index, entity_index])
    ordered_ids = ([] if equipped not in stacks else [equipped]) + sorted(
        (item for item in stacks if item != equipped),
        key=lambda item: asset_by_semantic_id.get(item, ""),
    )
    if len(ordered_ids) > 16:
        raise ValueError("authored native training loadout exceeds hotbar capacity")
    result: list[tuple[int, str, int]] = []
    for slot, item in enumerate(ordered_ids):
        try:
            asset = asset_by_semantic_id[item]
        except KeyError as error:
            raise ValueError(
                f"no pinned native asset name for semantic item ID {item}"
            ) from error
        result.append((slot, asset, stacks[item]))
    return tuple(result)


__all__ = [
    "native_training_hotbar_items_from_config",
    "training_inventory_from_config",
]
