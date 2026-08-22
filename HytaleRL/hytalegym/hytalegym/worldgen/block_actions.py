"""Portable block-tool and deterministic drop semantics for Hytale 0.5.7."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import struct
from typing import TYPE_CHECKING

from hytalegym.worldgen.block_affordances import gather_type_index


if TYPE_CHECKING:
    from hytalegym.worldgen.drop_programs import LocalDropProgram


BLOCK_ACTION_SCHEMA = "hytalerl_block_action_world_boundary_v4"
BLOCK_ACTION_VERSION = 4
BLOCK_TOOL_SPEC_CAPACITY = 16
BLOCK_TOOL_ROUTE_CAPACITY = 1
BLOCK_DROP_OUTPUT_CAPACITY = 10
BLOCK_DROP_METADATA_HASH_WORDS = 2
BLOCK_SEMANTIC_KEY_WORDS = hashlib.sha256().digest_size // struct.calcsize(">I")
_BLOCK_SEMANTIC_KEY_FORMAT = f">{BLOCK_SEMANTIC_KEY_WORDS}I"

SUPPORT_DROP_NONE = 0
SUPPORT_DROP_BREAK = 1
SUPPORT_DROP_DESTROY = 2
SUPPORT_DROP_FALL = 3
SUPPORT_DROP_TYPES = ("none", "break", "destroy", "fall")


@dataclass(frozen=True)
class LocalDropRoute:
    """One route accepted by ``BlockHarvestUtils.getDrops``."""

    available: bool
    randomized: bool
    item_asset_id: str | None
    item_id: int
    quantity: int
    item_max_stack: int
    durability: float
    max_durability: float
    metadata_hash: tuple[int, ...]
    program_sha256: str | None


@dataclass(frozen=True)
class LocalToolSpec:
    gather_type_index: int
    quality: int
    power: float


@dataclass(frozen=True)
class LocalHeldItemTool:
    """Held-item classification used by native block tool selection."""

    valid: bool
    weapon: bool
    builder_tool: bool
    tool_present: bool
    specs: tuple[LocalToolSpec, ...]


@dataclass(frozen=True)
class LocalBlockToolRoute:
    """One exact ``Gathering.Tools`` route keyed by interaction Tool."""

    present: bool
    tool_type: str | None
    tool_type_id: int
    replacement_semantic_key_valid: bool
    replacement_semantic_key: tuple[int, ...]
    drop: LocalDropRoute


@dataclass(frozen=True)
class LocalBlockActionSemantics:
    """Resolved block routes; runtime permission remains separate evidence."""

    semantic_key: tuple[int, ...]
    support_dependent: bool
    support_drop_type: int
    tool_breakable: bool
    soft_breakable: bool
    soft_weapon_breakable: bool
    harvestable: bool
    interaction_tool_route: LocalBlockToolRoute
    tool_drop: LocalDropRoute
    soft_drop: LocalDropRoute
    harvest_drop: LocalDropRoute


ItemLookup = Callable[[str], Mapping[str, object]]
DropProgramLookup = Callable[
    [str | Mapping[str, object]],
    "LocalDropProgram",
]


def stable_asset_id(asset_id: str) -> int:
    """Match the repository's positive FNV-1a host asset identifier."""

    value = 0x811C9DC5
    for byte in _asset_id(asset_id).encode("utf-8"):
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value & 0x7FFFFFFF or 1


def block_semantic_key(asset_id: str, rotation_index: int) -> tuple[int, ...]:
    """Match the staged native mutable-block SHA-256 key exactly."""

    if isinstance(rotation_index, bool) or not isinstance(rotation_index, int):
        raise TypeError("rotation_index must be an integer")
    if rotation_index < 0:
        raise ValueError("rotation_index must be nonnegative")
    payload = f"{_asset_id(asset_id)}\0{rotation_index}".encode("utf-8")
    return struct.unpack(
        _BLOCK_SEMANTIC_KEY_FORMAT,
        hashlib.sha256(payload).digest(),
    )


def block_asset_key(asset_id: str) -> tuple[int, ...]:
    """Return the stable asset-only key used for filler-root comparison."""

    return struct.unpack(
        _BLOCK_SEMANTIC_KEY_FORMAT,
        hashlib.sha256(_asset_id(asset_id).encode("utf-8")).digest(),
    )


def resolve_local_held_item_tool(
    item: Mapping[str, object] | None,
) -> LocalHeldItemTool:
    """Compile inherited item data into the fixed native tool predicate."""

    if item is None:
        return LocalHeldItemTool(True, False, False, False, ())
    if not isinstance(item, Mapping):
        raise TypeError("held item must be an object or None")
    weapon = item.get("Weapon") is not None
    builder = item.get("BuilderTool") is not None
    raw_tool = item.get("Tool")
    if raw_tool is None:
        return LocalHeldItemTool(True, weapon, builder, False, ())
    tool = _mapping(raw_tool, "Item.Tool")
    raw_specs = tool.get("Specs", [])
    if not isinstance(raw_specs, list):
        raise ValueError("Item.Tool.Specs must be an array")
    if len(raw_specs) > BLOCK_TOOL_SPEC_CAPACITY:
        raise ValueError("item tool spec capacity exceeded")
    specs = tuple(_tool_spec(value) for value in raw_specs)
    if len({spec.gather_type_index for spec in specs}) != len(specs):
        raise ValueError("item tool contains duplicate gather types")
    return LocalHeldItemTool(True, weapon, builder, True, specs)


def resolve_default_gather_spec(
    gather_type: str,
    value: Mapping[str, object] | None,
) -> LocalToolSpec | None:
    """Resolve one ``Item/Unarmed/Gathering`` asset by filename ID."""

    index = gather_type_index(gather_type)
    if value is None:
        return None
    data = _mapping(value, "unarmed gather spec")
    declared = data.get("GatherType", gather_type)
    if declared != gather_type:
        raise ValueError("unarmed gather spec ID and GatherType differ")
    return LocalToolSpec(
        gather_type_index=index,
        quality=_nonnegative_int16(data.get("Quality", 0), "gather Quality"),
        power=_positive_number(data.get("Power"), "gather Power"),
    )


def resolve_local_block_action_semantics(
    asset_id: str,
    item: Mapping[str, object],
    block: Mapping[str, object],
    *,
    rotation_index: int = 0,
    item_lookup: ItemLookup | None = None,
    drop_program_lookup: DropProgramLookup | None = None,
) -> LocalBlockActionSemantics:
    """Resolve exact tool routes and deterministic drops from inherited data."""

    identifier = _asset_id(asset_id)
    item_data = _mapping(item, "item")
    block_data = _mapping(block, "BlockType")
    gathering_value = block_data.get("Gathering")
    gathering = (
        None
        if gathering_value is None
        else _mapping(gathering_value, "BlockType.Gathering")
    )
    breaking = _optional_mapping(gathering, "Breaking")
    soft = _optional_mapping(gathering, "Soft")
    harvest = _optional_mapping(gathering, "Harvest")
    interaction_tool_route = _interaction_tool_route(
        identifier,
        item_data,
        block_data,
        gathering,
        item_lookup,
        drop_program_lookup,
    )

    gather_type = None if breaking is None else breaking.get("GatherType")
    if gather_type is not None and not isinstance(gather_type, str):
        raise ValueError("Breaking.GatherType must be a string")
    tool_breakable = (
        breaking is not None
        and gather_type is not None
        and gather_type != "Unbreakable"
    )
    soft_breakable = soft is not None
    soft_weapon_breakable = (
        False
        if soft is None
        else _boolean(
            soft.get("IsWeaponBreakable", True),
            "Soft.IsWeaponBreakable",
        )
    )
    quantity = (
        0
        if not tool_breakable
        else _nonnegative_int(
            breaking.get("Quantity", 1),
            "Breaking.Quantity",
        )
    )
    support_dependent, support_drop_type = _support_semantics(block_data)
    return LocalBlockActionSemantics(
        semantic_key=block_semantic_key(identifier, rotation_index),
        support_dependent=support_dependent,
        support_drop_type=support_drop_type,
        tool_breakable=tool_breakable,
        soft_breakable=soft_breakable,
        soft_weapon_breakable=soft_weapon_breakable,
        harvestable=harvest is not None,
        interaction_tool_route=interaction_tool_route,
        tool_drop=_drop_route(
            identifier,
            item_data,
            breaking if tool_breakable else None,
            quantity,
            item_lookup,
            drop_program_lookup,
        ),
        soft_drop=_drop_route(
            identifier,
            item_data,
            soft,
            1 if soft_breakable else 0,
            item_lookup,
            drop_program_lookup,
        ),
        harvest_drop=_drop_route(
            identifier,
            item_data,
            harvest,
            1 if harvest is not None else 0,
            item_lookup,
            drop_program_lookup,
        ),
    )


def block_action_contract() -> dict[str, object]:
    """Return the source-backed neutral World-to-consumer boundary."""

    from hytalegym.worldgen.drop_programs import (
        block_drop_program_contract_sha256,
    )

    return {
        "schema": BLOCK_ACTION_SCHEMA,
        "version": BLOCK_ACTION_VERSION,
        "server_version": "0.5.7",
        "identity": {
            "block": "sha256_words_of_utf8_asset_id_nul_decimal_rotation",
            "block_asset": "sha256_words_of_utf8_asset_id",
            "item": "positive_31bit_fnv1a_of_stable_asset_id",
            "runtime_ordinals": "never_published",
        },
        "tool": {
            "spec_capacity": BLOCK_TOOL_SPEC_CAPACITY,
            "installed_0_5_7_maximum": 14,
            "selection": (
                "matching_gather_type_then_quality_at_least_required"
            ),
            "no_tool": "ItemToolSpec_asset_for_target_gather_type",
            "weapon_or_builder": "excluded_from_tool_route",
            "soft_fallback": (
                "power_one_weapon_requires_Soft_IsWeaponBreakable"
            ),
            "interaction_route": {
                "capacity": BLOCK_TOOL_ROUTE_CAPACITY,
                "installed_0_5_7_maximum": 1,
                "key": (
                    "positive_31bit_fnv1a_of_exact_"
                    "BreakBlockInteraction_Tool_string"
                ),
                "match_tool": (
                    "reject_before_damage_when_exact_Gathering_Tools_Type_"
                    "route_is_absent"
                ),
                "matched_state": (
                    "replace_with_BlockType_getBlockForState_at_rotation_zero_"
                    "when_different_from_target"
                ),
                "matched_without_state": (
                    "break_and_suppress_default_drop_then_emit_route_drop"
                ),
            },
        },
        "drops": {
            "output_capacity": BLOCK_DROP_OUTPUT_CAPACITY,
            "installed_0_5_7_droplist_maximum": 10,
            "deterministic": "direct_ItemId_or_default_block_item",
            "named_or_embedded_droplist": (
                "exact_bounded_output_distribution_when_program_compiler_"
                "is_supplied_otherwise_unavailable"
            ),
            "drop_program_contract_sha256": (
                block_drop_program_contract_sha256()
            ),
            "partial_random_output": "forbidden",
            "item_stack": {
                "durability": "new_ItemStack_uses_Item_MaxDurability",
                "max_durability": "Item_MaxDurability_or_zero",
                "metadata_hash_words": BLOCK_DROP_METADATA_HASH_WORDS,
                "metadata": "canonical_zero_for_direct_or_default_item",
            },
        },
        "permissions": {
            "environment_block_modification": (
                "runtime_evidence_required_separately"
            ),
            "place": (
                "explicit_acceptance_evidence_required_never_inferred"
            ),
        },
        "removal_cascade": {
            "dependency": (
                "BlockType.hasSupport_equivalent_nonempty_Support_or_"
                "positive_MaxSupportDistance"
            ),
            "drop_type": {
                str(SUPPORT_DROP_NONE): "not_support_dependent",
                str(SUPPORT_DROP_BREAK): "BREAK",
                str(SUPPORT_DROP_DESTROY): "DESTROY",
                str(SUPPORT_DROP_FALL): "FALL",
            },
            "positive_certificate": (
                "single_cell_target_all_26_BlockFace_neighbours_known_"
                "and_none_support_dependent"
            ),
            "support_dependent_or_unknown": "fail_closed",
            "exact_cascade_simulation": False,
        },
        "provenance": "installed_assets_and_0_5_7_server_source",
    }


def block_action_contract_sha256() -> str:
    payload = json.dumps(
        block_action_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _interaction_tool_route(
    block_asset_id: str,
    block_item: Mapping[str, object],
    block: Mapping[str, object],
    gathering: Mapping[str, object] | None,
    item_lookup: ItemLookup | None,
    drop_program_lookup: DropProgramLookup | None,
) -> LocalBlockToolRoute:
    raw = None if gathering is None else gathering.get("Tools")
    if raw is None:
        return _empty_interaction_tool_route()
    if not isinstance(raw, list):
        raise ValueError("Gathering.Tools must be an array")
    if len(raw) > BLOCK_TOOL_ROUTE_CAPACITY:
        raise ValueError("Gathering.Tools route capacity exceeded")
    if not raw:
        return _empty_interaction_tool_route()
    route = _mapping(raw[0], "Gathering.Tools route")
    tool_type = _asset_id(route.get("Type"))
    state_value = route.get("State")
    replacement_valid = state_value not in (None, "", "Empty")
    replacement = (0,) * BLOCK_SEMANTIC_KEY_WORDS
    if replacement_valid:
        state_id = _asset_id(state_value)
        root = _root_block_asset_id(block_asset_id)
        if state_id == "default":
            reference = root
        else:
            state = block.get("State")
            definitions = (
                None
                if not isinstance(state, Mapping)
                else state.get("Definitions")
            )
            if (
                not isinstance(definitions, Mapping)
                or state_id not in definitions
            ):
                raise ValueError(
                    "Gathering.Tools.State is not defined by the block"
                )
            reference = f"*{root}_State_Definitions_{state_id}"
        # BlockAccessor.setBlock(BlockType) uses rotation zero, matching
        # BlockHarvestUtils' state-replacement branch.
        replacement = block_semantic_key(reference, 0)
    return LocalBlockToolRoute(
        present=True,
        tool_type=tool_type,
        tool_type_id=stable_asset_id(tool_type),
        replacement_semantic_key_valid=replacement_valid,
        replacement_semantic_key=replacement,
        drop=_drop_route(
            _root_block_asset_id(block_asset_id),
            block_item,
            route,
            1,
            item_lookup,
            drop_program_lookup,
        ),
    )


def _drop_route(
    block_asset_id: str,
    block_item: Mapping[str, object],
    route: Mapping[str, object] | None,
    quantity: int,
    item_lookup: ItemLookup | None,
    drop_program_lookup: DropProgramLookup | None,
) -> LocalDropRoute:
    if route is None:
        return _unavailable_drop_route()
    drop_list = route.get("DropList")
    item_value = route.get("ItemId")
    item_id = (
        None
        if item_value in (None, "", "Empty")
        else _asset_id(item_value)
    )
    if drop_list not in (None, "", "Empty"):
        if drop_program_lookup is not None:
            from hytalegym.worldgen.drop_programs import LocalDropProgram

            program = drop_program_lookup(drop_list)
            if not isinstance(program, LocalDropProgram):
                raise TypeError(
                    "drop_program_lookup must return LocalDropProgram"
                )
            return LocalDropRoute(
                True,
                program.randomized,
                None,
                0,
                0,
                0,
                0.0,
                0.0,
                (0,) * BLOCK_DROP_METADATA_HASH_WORDS,
                program.semantic_sha256,
            )
        program = json.dumps(
            drop_list,
            sort_keys=True,
            separators=(",", ":"),
        )
        return LocalDropRoute(
            False,
            True,
            None,
            0,
            0,
            0,
            0.0,
            0.0,
            (0,) * BLOCK_DROP_METADATA_HASH_WORDS,
            hashlib.sha256(program.encode("utf-8")).hexdigest(),
        )
    if quantity == 0:
        return LocalDropRoute(
            True,
            False,
            None,
            0,
            0,
            0,
            0.0,
            0.0,
            (0,) * BLOCK_DROP_METADATA_HASH_WORDS,
            None,
        )
    drop_asset_id = block_asset_id if item_id is None else item_id
    if drop_asset_id == block_asset_id:
        drop_item = block_item
    elif item_lookup is None:
        return _unavailable_drop_route()
    else:
        drop_item = _mapping(item_lookup(drop_asset_id), "drop item")
    durability = _item_max_durability(drop_item)
    return LocalDropRoute(
        True,
        False,
        drop_asset_id,
        stable_asset_id(drop_asset_id),
        quantity,
        _item_max_stack(drop_item),
        durability,
        durability,
        (0,) * BLOCK_DROP_METADATA_HASH_WORDS,
        None,
    )


def _tool_spec(value: object) -> LocalToolSpec:
    spec = _mapping(value, "Item.Tool.Spec")
    return LocalToolSpec(
        gather_type_index=gather_type_index(
            _asset_id(spec.get("GatherType"))
        ),
        quality=_nonnegative_int16(spec.get("Quality", 0), "tool Quality"),
        power=_positive_number(spec.get("Power"), "tool Power"),
    )


def _item_max_stack(item: Mapping[str, object]) -> int:
    authored = item.get("MaxStack")
    if authored is not None:
        return _positive_int(authored, "Item.MaxStack")
    unique = any(
        item.get(key) is not None
        for key in (
            "Tool",
            "Weapon",
            "Armor",
            "BuilderTool",
            "BlockSelectorTool",
        )
    )
    return 1 if unique else 100


def _item_max_durability(item: Mapping[str, object]) -> float:
    value = item.get("MaxDurability", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Item.MaxDurability must be numeric")
    result = float(value)
    if not 0.0 <= result < float("inf"):
        raise ValueError("Item.MaxDurability must be finite and nonnegative")
    return result


def _unavailable_drop_route() -> LocalDropRoute:
    return LocalDropRoute(
        False,
        False,
        None,
        0,
        0,
        0,
        0.0,
        0.0,
        (0,) * BLOCK_DROP_METADATA_HASH_WORDS,
        None,
    )


def _empty_interaction_tool_route() -> LocalBlockToolRoute:
    return LocalBlockToolRoute(
        present=False,
        tool_type=None,
        tool_type_id=0,
        replacement_semantic_key_valid=False,
        replacement_semantic_key=(0,) * BLOCK_SEMANTIC_KEY_WORDS,
        drop=_unavailable_drop_route(),
    )


def _root_block_asset_id(reference: str) -> str:
    identifier = _asset_id(reference)
    if not identifier.startswith("*"):
        return identifier
    encoded = identifier[1:]
    root, separator, state = encoded.partition("_State_Definitions_")
    if not separator or not root or not state:
        raise ValueError("malformed block state asset reference")
    return _asset_id(root)


def _optional_mapping(
    parent: Mapping[str, object] | None,
    key: str,
) -> Mapping[str, object] | None:
    if parent is None or parent.get(key) is None:
        return None
    return _mapping(parent[key], f"Gathering.{key}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _support_semantics(block: Mapping[str, object]) -> tuple[bool, int]:
    support = block.get("Support")
    if support is not None and not isinstance(support, Mapping):
        raise ValueError("BlockType.Support must be an object or null")
    if (
        support is None
        and block.get("Material", "Empty") == "Empty"
        and block.get("Group") != "@Tech"
    ):
        # BlockType.afterDecode() installs REQUIRED_BOTTOM_FACE_SUPPORT for
        # empty, non-technical blocks before hasSupport() is queried.
        support = {"Down": ({"FaceType": "Full"},)}
    maximum = _nonnegative_int(
        block.get("MaxSupportDistance", 0),
        "BlockType.MaxSupportDistance",
    )
    dependent = bool(support) or maximum > 0
    raw_drop = block.get("SupportDropType", "BREAK")
    if not isinstance(raw_drop, str) or not raw_drop:
        raise ValueError("BlockType.SupportDropType must be a string")
    drop_type = {
        "break": SUPPORT_DROP_BREAK,
        "destroy": SUPPORT_DROP_DESTROY,
        "fall": SUPPORT_DROP_FALL,
    }.get(raw_drop.casefold())
    if drop_type is None:
        raise ValueError("BlockType.SupportDropType is unknown")
    return dependent, drop_type if dependent else SUPPORT_DROP_NONE


def _asset_id(value: object) -> str:
    if not isinstance(value, str) or not value or any(
        character in value for character in "/\\\0"
    ):
        raise ValueError("asset ID must be a nonempty safe string")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_int16(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result > 32_767:
        raise ValueError(f"{label} must fit int16")
    return result


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not 0.0 < result < float("inf"):
        raise ValueError(f"{label} must be finite and positive")
    return result


__all__ = [
    "BLOCK_ACTION_SCHEMA",
    "BLOCK_ACTION_VERSION",
    "BLOCK_DROP_METADATA_HASH_WORDS",
    "BLOCK_DROP_OUTPUT_CAPACITY",
    "BLOCK_SEMANTIC_KEY_WORDS",
    "BLOCK_TOOL_ROUTE_CAPACITY",
    "BLOCK_TOOL_SPEC_CAPACITY",
    "SUPPORT_DROP_BREAK",
    "SUPPORT_DROP_DESTROY",
    "SUPPORT_DROP_FALL",
    "SUPPORT_DROP_NONE",
    "SUPPORT_DROP_TYPES",
    "LocalBlockActionSemantics",
    "LocalBlockToolRoute",
    "LocalDropRoute",
    "LocalHeldItemTool",
    "LocalToolSpec",
    "block_action_contract",
    "block_action_contract_sha256",
    "block_asset_key",
    "block_semantic_key",
    "resolve_default_gather_spec",
    "resolve_local_block_action_semantics",
    "resolve_local_held_item_tool",
    "stable_asset_id",
]
