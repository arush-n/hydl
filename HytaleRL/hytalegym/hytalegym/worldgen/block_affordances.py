"""Stable Hytale 0.5.7 block-affordance vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping


BLOCK_AFFORDANCE_SCHEMA = "hytalerl_block_affordance_dictionary_v1"
BLOCK_AFFORDANCE_VERSION = 1

# Bit positions are checkpoint ABI. Keep descriptive asset tags out of this
# policy vocabulary unless they change an available world interaction.
BLOCK_AFFORDANCE_TAGS = (
    "breakable",
    "ore",
    "harvestable",
    "soft",
    "has_break_drop",
    "use_default_drop_when_placed",
    "door",
    "stateful",
)
BLOCK_AFFORDANCE_TAG_CAPACITY = 16

# Index zero is the valid absence of a Breaking declaration. Unknown strings
# are rejected rather than assigned a process-dependent runtime ordinal.
GATHER_TYPES = (
    "none",
    "Benches",
    "Branches",
    "Cloths",
    "DungeonBlocks",
    "OreAdamantite",
    "OreCobalt",
    "OreCopper",
    "OreGold",
    "OreIron",
    "OreMithril",
    "OreSilver",
    "OreThorium",
    "Pickaxe_Tier0",
    "Rocks",
    "SoftBlocks",
    "SoftWoods",
    "Soils",
    "Unbreakable",
    "VolcanicRocks",
    "Woods",
)
GATHER_TYPE_NONE = 0
GATHER_TYPE_UNBREAKABLE = GATHER_TYPES.index("Unbreakable")


@dataclass(frozen=True)
class BlockAffordance:
    """One portable, policy-meaningful block affordance value."""

    valid: bool
    tags: int
    gather_type_index: int
    required_tool_quality: int


def block_affordance_tag_mask(*tags: str) -> int:
    """Encode known policy tags, rejecting aliases and unknown values."""

    if len(set(tags)) != len(tags):
        raise ValueError("block affordance tags must be unique")
    indices = {name: index for index, name in enumerate(BLOCK_AFFORDANCE_TAGS)}
    unknown = sorted(set(tags) - indices.keys())
    if unknown:
        raise ValueError(f"unknown block affordance tags: {unknown}")
    return sum(1 << indices[tag] for tag in tags)


def gather_type_index(value: str | None) -> int:
    """Return the portable gather-type index or fail closed."""

    canonical = "none" if value is None else value
    if not isinstance(canonical, str):
        raise TypeError("gather type must be a string or None")
    try:
        return GATHER_TYPES.index(canonical)
    except ValueError as error:
        raise ValueError(f"unknown gather type: {canonical!r}") from error


def resolve_local_block_affordance(
    item: Mapping[str, object],
    block: Mapping[str, object],
) -> BlockAffordance:
    """Resolve inherited local asset data into the portable vocabulary."""

    gathering = block.get("Gathering")
    if gathering is not None and not isinstance(gathering, Mapping):
        raise ValueError("BlockType.Gathering must be an object")
    breaking = gathering.get("Breaking") if gathering is not None else None
    if breaking is not None and not isinstance(breaking, Mapping):
        raise ValueError("BlockType.Gathering.Breaking must be an object")
    soft = gathering.get("Soft") if gathering is not None else None
    if soft is not None and not isinstance(soft, Mapping):
        raise ValueError("BlockType.Gathering.Soft must be an object")

    gather_type = None if breaking is None else breaking.get("GatherType")
    if gather_type is not None and not isinstance(gather_type, str):
        raise ValueError("block GatherType must be a string")
    gather_index = gather_type_index(gather_type)
    quality = 0 if breaking is None else breaking.get("Quality", 0)
    if (
        isinstance(quality, bool)
        or not isinstance(quality, int)
        or not 0 <= quality <= 32_767
    ):
        raise ValueError("block gathering Quality must fit nonnegative int16")
    quantity = 0 if breaking is None else breaking.get("Quantity", 1)
    if (
        isinstance(quantity, bool)
        or not isinstance(quantity, int)
        or quantity < 0
    ):
        raise ValueError("block gathering Quantity must be nonnegative int")

    tags: list[str] = []
    tool_breakable = (
        breaking is not None
        and gather_type is not None
        and gather_type != "Unbreakable"
    )
    soft_breakable = soft is not None
    if tool_breakable or soft_breakable:
        tags.append("breakable")
    categories = item.get("Categories", [])
    if (
        not isinstance(categories, list)
        or not all(isinstance(value, str) for value in categories)
    ):
        raise ValueError("item Categories must be an array of strings")
    if (
        "Blocks.Ores" in categories
        or (gather_type or "").startswith("Ore")
    ):
        tags.append("ore")
    if gathering is not None and gathering.get("Harvest") is not None:
        tags.append("harvestable")
    if soft_breakable:
        tags.append("soft")
    tool_drop = tool_breakable and quantity > 0 and (
        any(
            _authored_drop(breaking.get(key))
            for key in ("ItemId", "DropList")
        )
        or bool(item)
    )
    soft_drop = soft_breakable and (
        any(
            _authored_drop(soft.get(key))
            for key in ("ItemId", "DropList")
        )
        or bool(item)
    )
    if tool_drop or soft_drop:
        tags.append("has_break_drop")
    if gathering is not None:
        default_drop = gathering.get("UseDefaultDropWhenPlaced", False)
        if not isinstance(default_drop, bool):
            raise ValueError(
                "BlockType.Gathering.UseDefaultDropWhenPlaced must be boolean"
            )
        if default_drop:
            tags.append("use_default_drop_when_placed")
    door = block.get("IsDoor", False)
    if not isinstance(door, bool):
        raise ValueError("BlockType.IsDoor must be boolean")
    if door:
        tags.append("door")
    if isinstance(block.get("State"), Mapping):
        tags.append("stateful")
    return BlockAffordance(
        valid=True,
        tags=block_affordance_tag_mask(*tags),
        gather_type_index=gather_index,
        required_tool_quality=quality,
    )


def _authored_drop(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value != "Empty"
    ) or (isinstance(value, Mapping) and bool(value))


def block_affordance_dictionary_contract() -> dict[str, object]:
    """Return the portable, checkpoint-pinnable affordance dictionary."""

    return {
        "schema": BLOCK_AFFORDANCE_SCHEMA,
        "version": BLOCK_AFFORDANCE_VERSION,
        "server_version": "0.5.7",
        "tag_encoding": {
            "dtype": "uint16",
            "capacity": BLOCK_AFFORDANCE_TAG_CAPACITY,
            "dictionary": {
                name: index
                for index, name in enumerate(BLOCK_AFFORDANCE_TAGS)
            },
            "unknown": "reject_complete_geometry_value",
        },
        "gather_type_encoding": {
            "dtype": "uint8",
            "dictionary": {
                name: index for index, name in enumerate(GATHER_TYPES)
            },
            "none": GATHER_TYPE_NONE,
            "unknown": "reject_complete_geometry_value",
        },
        "required_tool_quality": {
            "dtype": "int16",
            "native_source": "BlockBreakingDropType.getQuality",
            "minimum": 0,
            "maximum": 32_767,
        },
        "derivation": {
            "breakable": (
                "native_tool_route_with_nonempty_non_Unbreakable_"
                "GatherType_or_Soft_route_present"
            ),
            "ore": (
                "item_category_Blocks.Ores_or_gather_type_prefix_Ore"
            ),
            "harvestable": "BlockGathering.isHarvestable",
            "soft": "BlockGathering.isSoft",
            "has_break_drop": (
                "positive_quantity_tool_route_or_Soft_route_with_authored_"
                "ItemId_embedded_or_named_DropList_or_default_block_item"
            ),
            "use_default_drop_when_placed": (
                "BlockGathering.shouldUseDefaultDropWhenPlaced"
            ),
            "door": "BlockType.isDoor",
            "stateful": "BlockType.getState_non_null_or_isState",
        },
        "non_policy_identity": (
            "canonical_asset_state_rotation_sha256_remains_separate"
        ),
        "raw_runtime_ordinals": "never_published",
    }


def block_affordance_dictionary_sha256() -> str:
    """Return the canonical dictionary digest."""

    payload = json.dumps(
        block_affordance_dictionary_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "BLOCK_AFFORDANCE_SCHEMA",
    "BLOCK_AFFORDANCE_TAG_CAPACITY",
    "BLOCK_AFFORDANCE_TAGS",
    "BLOCK_AFFORDANCE_VERSION",
    "GATHER_TYPES",
    "GATHER_TYPE_NONE",
    "GATHER_TYPE_UNBREAKABLE",
    "BlockAffordance",
    "block_affordance_dictionary_contract",
    "block_affordance_dictionary_sha256",
    "block_affordance_tag_mask",
    "gather_type_index",
    "resolve_local_block_affordance",
]
