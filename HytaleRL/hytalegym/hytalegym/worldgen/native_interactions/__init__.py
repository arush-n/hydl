"""Typed native evidence for resolved item triggers and interaction chains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# Re-exported to preserve this module's attribute surface.
from hytalegym.worldgen.native_interactions._contract import (  # noqa: F401
    _canonical_sha256,
    _native_item_interaction_contract,
    _sha256,
)
from hytalegym.worldgen.native_interactions._scalars import (  # noqa: F401
    _array,
    _boolean,
    _finite_number,
    _integer,
    _jsonable,
    _mapping,
    _number,
    _pair,
    _require_exact_fields,
    _string,
    _string_tuple,
)


NATIVE_ITEM_INTERACTION_SCHEMA_V2 = (
    "hytalerl_native_item_interaction_evidence_v2"
)
NATIVE_ITEM_INTERACTION_SCHEMA_V3 = (
    "hytalerl_native_item_interaction_evidence_v3"
)
NATIVE_ITEM_INTERACTION_SCHEMA = NATIVE_ITEM_INTERACTION_SCHEMA_V2
NATIVE_ITEM_INTERACTION_VERSION = 2
NATIVE_ITEM_INTERACTION_VERSION_V3 = 3
NATIVE_ITEM_TRIGGER_CAPACITY = 25
NATIVE_ITEM_INTERACTION_CAPACITY = 256
NATIVE_ITEM_EDGE_CAPACITY = 512
NATIVE_ITEM_CHARGE_TIME_CAPACITY = 16
NATIVE_ITEM_BLOCK_CHANGE_CAPACITY = 256
NATIVE_ITEM_METADATA_CAPACITY = 16_384

INTERACTION_TYPE_NAMES = (
    "Primary",
    "Secondary",
    "Ability1",
    "Ability2",
    "Ability3",
    "Use",
    "Pick",
    "Pickup",
    "CollisionEnter",
    "CollisionLeave",
    "Collision",
    "EntityStatEffect",
    "SwapTo",
    "SwapFrom",
    "Death",
    "Wielding",
    "ProjectileSpawn",
    "ProjectileHit",
    "ProjectileMiss",
    "ProjectileBounce",
    "Held",
    "HeldOffhand",
    "Equipped",
    "Dodge",
    "GameModeSwap",
)


@dataclass(frozen=True, slots=True)
class NativeInteractionRules:
    blocked_by_mask: int
    blocked_by_uses_default: bool
    blocking_mask: int
    interrupted_by_mask: int
    interrupted_by_unspecified: bool
    interrupting_mask: int
    interrupting_unspecified: bool
    blocked_by_bypass_index: int
    blocking_bypass_index: int
    interrupted_by_bypass_index: int
    interrupting_bypass_index: int

    @classmethod
    def from_response(
        cls,
        value: Mapping[str, Any],
    ) -> NativeInteractionRules:
        masks = tuple(
            _integer(value, key)
            for key in (
                "blocked_by_mask",
                "blocking_mask",
                "interrupted_by_mask",
                "interrupting_mask",
            )
        )
        if any(mask < 0 or mask >= (1 << NATIVE_ITEM_TRIGGER_CAPACITY)
               for mask in masks):
            raise ValueError("interaction-rule mask is outside its domain")
        return cls(
            blocked_by_mask=masks[0],
            blocked_by_uses_default=_boolean(
                value,
                "blocked_by_uses_default",
            ),
            blocking_mask=masks[1],
            interrupted_by_mask=masks[2],
            interrupted_by_unspecified=_boolean(
                value,
                "interrupted_by_unspecified",
            ),
            interrupting_mask=masks[3],
            interrupting_unspecified=_boolean(
                value,
                "interrupting_unspecified",
            ),
            blocked_by_bypass_index=_integer(
                value,
                "blocked_by_bypass_index",
            ),
            blocking_bypass_index=_integer(
                value,
                "blocking_bypass_index",
            ),
            interrupted_by_bypass_index=_integer(
                value,
                "interrupted_by_bypass_index",
            ),
            interrupting_bypass_index=_integer(
                value,
                "interrupting_bypass_index",
            ),
        )


@dataclass(frozen=True, slots=True)
class NativeInteractionModeSettings:
    game_mode: int
    allow_skip_chain_on_click: bool
    cooldown_id: str
    cooldown_seconds: float
    click_bypass: bool
    skip_cooldown_reset: bool
    interrupt_recharge: bool
    charge_times: tuple[float, ...]

    @classmethod
    def from_response(
        cls,
        value: Mapping[str, Any],
    ) -> NativeInteractionModeSettings:
        game_mode = _integer(value, "game_mode")
        cooldown = _number(value, "cooldown_seconds")
        charges = tuple(
            _finite_number(item, "charge_times")
            for item in _array(value, "charge_times")
        )
        if (
            game_mode not in (0, 1)
            or cooldown < 0.0
            or not 1 <= len(charges) <= NATIVE_ITEM_CHARGE_TIME_CAPACITY
            or any(charge < 0.0 for charge in charges)
        ):
            raise ValueError("interaction mode settings exceed capacity")
        return cls(
            game_mode=game_mode,
            allow_skip_chain_on_click=_boolean(
                value,
                "allow_skip_chain_on_click",
            ),
            cooldown_id=_string(value, "cooldown_id"),
            cooldown_seconds=cooldown,
            click_bypass=_boolean(value, "click_bypass"),
            skip_cooldown_reset=_boolean(
                value,
                "skip_cooldown_reset",
            ),
            interrupt_recharge=_boolean(value, "interrupt_recharge"),
            charge_times=charges,
        )


@dataclass(frozen=True, slots=True)
class NativeInteractionItemPayload:
    item_asset_id: str
    quantity: int
    durability: float
    max_durability: float
    override_dropped_item_animation: bool
    metadata_json: str

    @classmethod
    def from_response(
        cls,
        value: Mapping[str, Any],
    ) -> NativeInteractionItemPayload:
        _require_exact_fields(
            value,
            {
                "item_asset_id",
                "quantity",
                "durability",
                "max_durability",
                "override_dropped_item_animation",
                "metadata_json",
            },
        )
        quantity = _integer(value, "quantity")
        metadata = _string(value, "metadata_json", allow_empty=True)
        if quantity <= 0 or len(metadata) > NATIVE_ITEM_METADATA_CAPACITY:
            raise ValueError("interaction item payload exceeds capacity")
        return cls(
            item_asset_id=_string(value, "item_asset_id"),
            quantity=quantity,
            durability=_number(value, "durability"),
            max_durability=_number(value, "max_durability"),
            override_dropped_item_animation=_boolean(
                value,
                "override_dropped_item_animation",
            ),
            metadata_json=metadata,
        )


@dataclass(frozen=True, slots=True)
class NativeBreakBlockPayload:
    """Authored server-only break-tool semantics.

    ``tool_id`` is a tool category consumed by native block-damage logic, not
    an item asset ID. ``match_tool`` makes that category an exact legality
    requirement.
    """

    tool_id: str
    match_tool: bool


@dataclass(frozen=True, slots=True)
class NativePlaceBlockPayload:
    block_asset_id: str
    remove_item_in_hand: bool
    allow_drag_placement: bool


@dataclass(frozen=True, slots=True)
class NativeChangeBlockPayload:
    changes: tuple[tuple[str, str], ...]
    world_sound_event_asset_id: str
    require_not_broken: bool


@dataclass(frozen=True, slots=True)
class NativeModifyInventoryPayload:
    required_game_mode: int
    item_to_remove: NativeInteractionItemPayload | None
    adjust_held_item_quantity: int
    item_to_add: NativeInteractionItemPayload | None
    broken_item_asset_id: str
    adjust_held_item_durability: float
    notify_on_break_specified: bool
    notify_on_break: bool
    notify_on_break_message: str


NativeInteractionPayload = (
    NativeBreakBlockPayload
    | NativePlaceBlockPayload
    | NativeChangeBlockPayload
    | NativeModifyInventoryPayload
    | None
)


@dataclass(frozen=True, slots=True)
class NativeInteractionNode:
    interaction_id: str
    implementation_class: str
    run_time: float
    cancel_on_item_change: bool
    wait_for_data_from: int
    next_interaction_id: str
    failed_interaction_id: str
    use_latest_target: bool
    harvest: bool
    horizontal_speed_multiplier: float
    effects_present: bool
    start_delay: float
    wait_for_animation_to_finish: bool
    movement_effects_present: bool
    movement_disable_all: bool
    movement_lock_mask: int
    payload: NativeInteractionPayload

    @classmethod
    def from_response(
        cls,
        value: Mapping[str, Any],
    ) -> NativeInteractionNode:
        _require_exact_fields(
            value,
            {
                "interaction_id",
                "implementation_class",
                "run_time",
                "cancel_on_item_change",
                "wait_for_data_from",
                "next_interaction_id",
                "failed_interaction_id",
                "use_latest_target",
                "harvest",
                "horizontal_speed_multiplier",
                "effects_present",
                "start_delay",
                "wait_for_animation_to_finish",
                "movement_effects_present",
                "movement_disable_all",
                "movement_lock_mask",
                "payload",
            },
        )
        run_time = _number(value, "run_time")
        wait = _integer(value, "wait_for_data_from")
        horizontal = _number(value, "horizontal_speed_multiplier")
        effects_present = _boolean(value, "effects_present")
        start_delay = _number(value, "start_delay")
        wait_for_animation = _boolean(
            value,
            "wait_for_animation_to_finish",
        )
        movement_present = _boolean(value, "movement_effects_present")
        movement_disable_all = _boolean(
            value,
            "movement_disable_all",
        )
        movement_mask = _integer(value, "movement_lock_mask")
        if (
            run_time < 0.0
            or wait not in (0, 1, 2)
            or start_delay < 0.0
            or movement_mask not in range(128)
            or (not movement_present and movement_mask != 0)
            or (
                movement_disable_all
                and (
                    not movement_present
                    or movement_mask != 0x7F
                )
            )
            or (
                not effects_present
                and (
                    start_delay != 0.0
                    or wait_for_animation
                    or movement_present
                )
            )
        ):
            raise ValueError("interaction node is outside its domain")
        return cls(
            interaction_id=_string(value, "interaction_id"),
            implementation_class=_string(value, "implementation_class"),
            run_time=run_time,
            cancel_on_item_change=_boolean(
                value,
                "cancel_on_item_change",
            ),
            wait_for_data_from=wait,
            next_interaction_id=_string(
                value,
                "next_interaction_id",
                allow_empty=True,
            ),
            failed_interaction_id=_string(
                value,
                "failed_interaction_id",
                allow_empty=True,
            ),
            use_latest_target=_boolean(value, "use_latest_target"),
            harvest=_boolean(value, "harvest"),
            horizontal_speed_multiplier=horizontal,
            effects_present=effects_present,
            start_delay=start_delay,
            wait_for_animation_to_finish=wait_for_animation,
            movement_effects_present=movement_present,
            movement_disable_all=movement_disable_all,
            movement_lock_mask=movement_mask,
            payload=_interaction_payload(
                _mapping(value.get("payload"))
            ),
        )


def _interaction_payload(
    value: Mapping[str, Any],
) -> NativeInteractionPayload:
    kind = _string(value, "kind", allow_empty=True)
    if not kind:
        _require_exact_fields(value, {"kind"})
        return None
    if kind == "break_block":
        _require_exact_fields(
            value,
            {
                "kind",
                "tool_id",
                "match_tool",
            },
        )
        tool_id = _string(value, "tool_id", allow_empty=True)
        match_tool = _boolean(value, "match_tool")
        if match_tool and not tool_id:
            raise ValueError(
                "break-block exact tool matching requires a tool ID"
            )
        return NativeBreakBlockPayload(
            tool_id=tool_id,
            match_tool=match_tool,
        )
    if kind == "place_block":
        _require_exact_fields(
            value,
            {
                "kind",
                "block_asset_id",
                "remove_item_in_hand",
                "allow_drag_placement",
            },
        )
        return NativePlaceBlockPayload(
            block_asset_id=_string(
                value,
                "block_asset_id",
                allow_empty=True,
            ),
            remove_item_in_hand=_boolean(value, "remove_item_in_hand"),
            allow_drag_placement=_boolean(value, "allow_drag_placement"),
        )
    if kind == "change_block":
        _require_exact_fields(
            value,
            {
                "kind",
                "changes",
                "world_sound_event_asset_id",
                "require_not_broken",
            },
        )
        changes = tuple(
            (
                _string({"value": _pair(item)[0]}, "value"),
                _string({"value": _pair(item)[1]}, "value"),
            )
            for item in _array(value, "changes")
        )
        if (
            len(changes) > NATIVE_ITEM_BLOCK_CHANGE_CAPACITY
            or changes != tuple(sorted(set(changes)))
        ):
            raise ValueError("block-change payload exceeds its capacity")
        return NativeChangeBlockPayload(
            changes=changes,
            world_sound_event_asset_id=_string(
                value,
                "world_sound_event_asset_id",
                allow_empty=True,
            ),
            require_not_broken=_boolean(value, "require_not_broken"),
        )
    if kind == "modify_inventory":
        _require_exact_fields(
            value,
            {
                "kind",
                "required_game_mode",
                "item_to_remove",
                "adjust_held_item_quantity",
                "item_to_add",
                "broken_item_asset_id",
                "adjust_held_item_durability",
                "notify_on_break_specified",
                "notify_on_break",
                "notify_on_break_message",
            },
        )
        mode = _integer(value, "required_game_mode")
        if mode not in (-1, 0, 1):
            raise ValueError("modify-inventory game mode is invalid")
        return NativeModifyInventoryPayload(
            required_game_mode=mode,
            item_to_remove=_optional_item_payload(
                value.get("item_to_remove")
            ),
            adjust_held_item_quantity=_integer(
                value,
                "adjust_held_item_quantity",
            ),
            item_to_add=_optional_item_payload(value.get("item_to_add")),
            broken_item_asset_id=_string(
                value,
                "broken_item_asset_id",
                allow_empty=True,
            ),
            adjust_held_item_durability=_number(
                value,
                "adjust_held_item_durability",
            ),
            notify_on_break_specified=_boolean(
                value,
                "notify_on_break_specified",
            ),
            notify_on_break=_boolean(value, "notify_on_break"),
            notify_on_break_message=_string(
                value,
                "notify_on_break_message",
                allow_empty=True,
            ),
        )
    raise ValueError(f"unsupported interaction payload kind: {kind}")


def _optional_item_payload(
    value: object,
) -> NativeInteractionItemPayload | None:
    return (
        None
        if value is None
        else NativeInteractionItemPayload.from_response(_mapping(value))
    )


@dataclass(frozen=True, slots=True)
class NativeInteractionEdge:
    parent_interaction_id: str
    child_interaction_id: str
    relation: str

    @classmethod
    def from_response(
        cls,
        value: Mapping[str, Any],
    ) -> NativeInteractionEdge:
        return cls(
            parent_interaction_id=_string(
                value,
                "parent_interaction_id",
                allow_empty=True,
            ),
            child_interaction_id=_string(
                value,
                "child_interaction_id",
            ),
            relation=_string(value, "relation"),
        )


@dataclass(frozen=True, slots=True)
class NativeInteractionRoot:
    root_id: str
    click_queuing_timeout: float
    require_new_click: bool
    rules: NativeInteractionRules
    mode_settings: tuple[NativeInteractionModeSettings, ...]
    initial_interaction_ids: tuple[str, ...]

    @classmethod
    def from_response(
        cls,
        value: Mapping[str, Any],
    ) -> NativeInteractionRoot:
        timeout = _number(value, "click_queuing_timeout")
        settings = tuple(
            NativeInteractionModeSettings.from_response(_mapping(item))
            for item in _array(value, "mode_settings")
        )
        initial = _string_tuple(value, "initial_interaction_ids")
        if (
            timeout < 0.0
            or tuple(row.game_mode for row in settings) != (0, 1)
            or not 1 <= len(initial) <= NATIVE_ITEM_INTERACTION_CAPACITY
        ):
            raise ValueError("interaction root is outside its capacity")
        return cls(
            root_id=_string(value, "root_id"),
            click_queuing_timeout=timeout,
            require_new_click=_boolean(value, "require_new_click"),
            rules=NativeInteractionRules.from_response(
                _mapping(value.get("rules"))
            ),
            mode_settings=settings,
            initial_interaction_ids=initial,
        )


@dataclass(frozen=True, slots=True)
class NativeInteractionTrigger:
    interaction_type: int
    interaction_type_name: str
    item_asset_id: str
    held_item_section_id: int
    held_item_slot: int
    root: NativeInteractionRoot | None
    nodes: tuple[NativeInteractionNode, ...]
    edges: tuple[NativeInteractionEdge, ...]

    @classmethod
    def from_response(
        cls,
        value: Mapping[str, Any],
    ) -> NativeInteractionTrigger:
        interaction_type = _integer(value, "interaction_type")
        root_present = _boolean(value, "root_present")
        raw_root = value.get("root")
        root = (
            NativeInteractionRoot.from_response(_mapping(raw_root))
            if root_present
            else None
        )
        if (raw_root is not None) != root_present:
            raise ValueError("interaction root presence is inconsistent")
        nodes = tuple(
            NativeInteractionNode.from_response(_mapping(item))
            for item in _array(value, "nodes")
        )
        edges = tuple(
            NativeInteractionEdge.from_response(_mapping(item))
            for item in _array(value, "edges")
        )
        if (
            not 0 <= interaction_type < NATIVE_ITEM_TRIGGER_CAPACITY
            or _integer(value, "node_count") != len(nodes)
            or _integer(value, "edge_count") != len(edges)
            or len(nodes) > NATIVE_ITEM_INTERACTION_CAPACITY
            or len(edges) > NATIVE_ITEM_EDGE_CAPACITY
            or (not root_present and (nodes or edges))
        ):
            raise ValueError("interaction trigger exceeds its capacity")
        node_ids = tuple(node.interaction_id for node in nodes)
        edge_keys = tuple(
            (
                edge.parent_interaction_id,
                edge.child_interaction_id,
                edge.relation,
            )
            for edge in edges
        )
        node_set = set(node_ids)
        references = tuple(
            reference
            for node in nodes
            for reference in (
                node.next_interaction_id,
                node.failed_interaction_id,
            )
            if reference
        )
        if (
            node_ids != tuple(sorted(set(node_ids)))
            or edge_keys != tuple(sorted(edge_keys))
            or any(
                edge.child_interaction_id not in node_set
                or (
                    edge.parent_interaction_id
                    and edge.parent_interaction_id not in node_set
                )
                for edge in edges
            )
            or any(reference not in node_set for reference in references)
            or (
                root is not None
                and any(
                    interaction not in node_set
                    for interaction in root.initial_interaction_ids
                )
            )
        ):
            raise ValueError("interaction graph is incomplete or unordered")
        return cls(
            interaction_type=interaction_type,
            interaction_type_name=_string(value, "interaction_type_name"),
            item_asset_id=_string(
                value,
                "item_asset_id",
                allow_empty=True,
            ),
            held_item_section_id=_integer(
                value,
                "held_item_section_id",
            ),
            held_item_slot=_integer(value, "held_item_slot"),
            root=root,
            nodes=nodes,
            edges=edges,
        )


@dataclass(frozen=True, slots=True)
class NativeItemInteractionEvidence:
    bridge_sha256: str
    server_version: str
    world: str
    worldgen_provider: str
    worldgen_version: str
    seed: int
    active_game_mode: int
    equipped_slot: int
    equipped_slot_capacity: int
    equipped_slot_available: bool
    triggers: tuple[NativeInteractionTrigger, ...]
    schema: str = NATIVE_ITEM_INTERACTION_SCHEMA
    version: int = NATIVE_ITEM_INTERACTION_VERSION

    @classmethod
    def from_response(
        cls,
        value: Mapping[str, Any],
        *,
        expected_bridge_sha256: str | None = None,
    ) -> NativeItemInteractionEvidence:
        if value.get("type") != "item_interaction_evidence":
            raise ValueError("bridge returned the wrong interaction type")
        schema = value.get("schema")
        version = value.get("version")
        if (
            (schema, version)
            not in {
                (
                    NATIVE_ITEM_INTERACTION_SCHEMA_V2,
                    NATIVE_ITEM_INTERACTION_VERSION,
                ),
                (
                    NATIVE_ITEM_INTERACTION_SCHEMA_V3,
                    NATIVE_ITEM_INTERACTION_VERSION_V3,
                ),
            }
        ):
            raise ValueError(
                "bridge returned an unsupported interaction schema/version"
            )
        bridge = _sha256(_string(value, "bridge_sha256"))
        if (
            expected_bridge_sha256 is not None
            and bridge.lower() != _sha256(expected_bridge_sha256).lower()
        ):
            raise ValueError(
                "interaction evidence came from a different bridge"
            )
        if (
            _integer(value, "expected_trigger_count")
            != NATIVE_ITEM_TRIGGER_CAPACITY
            or _integer(value, "trigger_count")
            != NATIVE_ITEM_TRIGGER_CAPACITY
            or _integer(value, "interaction_capacity_per_trigger")
            != NATIVE_ITEM_INTERACTION_CAPACITY
            or _integer(value, "edge_capacity_per_trigger")
            != NATIVE_ITEM_EDGE_CAPACITY
            or _integer(value, "charge_time_capacity")
            != NATIVE_ITEM_CHARGE_TIME_CAPACITY
            or _integer(value, "block_change_capacity_per_node")
            != NATIVE_ITEM_BLOCK_CHANGE_CAPACITY
            or _integer(value, "item_metadata_capacity")
            != NATIVE_ITEM_METADATA_CAPACITY
            or value.get("evidence_scope")
            != (
                "resolved_item_roots_rules_effects_and_"
                "reachable_authored_chain"
            )
            or value.get("public_player_acceptance_certified") is not False
        ):
            raise ValueError("native interaction capability boundary changed")
        triggers = tuple(
            NativeInteractionTrigger.from_response(_mapping(item))
            for item in _array(value, "triggers")
        )
        if (
            tuple(row.interaction_type for row in triggers)
            != tuple(range(NATIVE_ITEM_TRIGGER_CAPACITY))
            or tuple(row.interaction_type_name for row in triggers)
            != INTERACTION_TYPE_NAMES
        ):
            raise ValueError("native interaction trigger vocabulary changed")
        if version >= NATIVE_ITEM_INTERACTION_VERSION_V3:
            break_nodes = tuple(
                node
                for trigger in triggers
                for node in trigger.nodes
                if node.implementation_class.rsplit(".", 1)[-1]
                == "BreakBlockInteraction"
            )
            if any(
                not isinstance(node.payload, NativeBreakBlockPayload)
                for node in break_nodes
            ):
                raise ValueError(
                    "v3 BreakBlockInteraction lacks typed tool semantics"
                )
        active_mode = _integer(value, "active_game_mode")
        equipped_slot = _integer(value, "equipped_slot")
        equipped_capacity = _integer(value, "equipped_slot_capacity")
        equipped_available = _boolean(value, "equipped_slot_available")
        if (
            active_mode not in (0, 1)
            or equipped_slot < 0
            or equipped_capacity < 0
            or equipped_available != (equipped_slot < equipped_capacity)
            or (
                not equipped_available
                and (equipped_capacity != 0 or equipped_slot != 0)
            )
        ):
            raise ValueError(
                "native interaction game mode or equipped slot is invalid"
            )
        return cls(
            bridge_sha256=bridge,
            server_version=_string(value, "server_version"),
            world=_string(value, "world"),
            worldgen_provider=_string(value, "worldgen_provider"),
            worldgen_version=_string(value, "worldgen_version"),
            seed=_integer(value, "seed"),
            active_game_mode=active_mode,
            equipped_slot=equipped_slot,
            equipped_slot_capacity=equipped_capacity,
            equipped_slot_available=equipped_available,
            triggers=triggers,
            schema=str(schema),
            version=int(version),
        )

    def semantic_sha256(self) -> str:
        """Hash resolved meaning while excluding bridge/world provenance."""

        return _canonical_sha256({
            "schema": self.schema,
            "version": self.version,
            "active_game_mode": self.active_game_mode,
            "equipped_slot": self.equipped_slot,
            "equipped_slot_capacity": self.equipped_slot_capacity,
            "equipped_slot_available": self.equipped_slot_available,
            "triggers": _jsonable(self.triggers),
        })


def native_item_interaction_evidence_request(
    equipped_slot: int = 0,
) -> dict[str, str | int]:
    if isinstance(equipped_slot, bool) or not isinstance(equipped_slot, int):
        raise TypeError("equipped_slot must be an integer")
    if equipped_slot < 0:
        raise ValueError("equipped_slot must be nonnegative")
    return {
        "type": "item_interaction_evidence",
        "equipped_slot": equipped_slot,
    }


def native_item_interaction_contract() -> dict[str, object]:
    return _native_item_interaction_contract(
        schema=NATIVE_ITEM_INTERACTION_SCHEMA,
        version=NATIVE_ITEM_INTERACTION_VERSION,
        resolved_subclass_payloads=(
            "typed_PlaceBlock_ChangeBlock_ModifyInventory_"
            "subclass_payloads"
        ),
    )


def native_item_interaction_v3_contract() -> dict[str, object]:
    """Return the staged v3 contract without moving the published v2 alias."""

    return _native_item_interaction_contract(
        schema=NATIVE_ITEM_INTERACTION_SCHEMA_V3,
        version=NATIVE_ITEM_INTERACTION_VERSION_V3,
        resolved_subclass_payloads=(
            "typed_BreakBlock_Tool_MatchTool_PlaceBlock_ChangeBlock_"
            "ModifyInventory_subclass_payloads"
        ),
    )


def native_item_interaction_contract_sha256() -> str:
    return _canonical_sha256(native_item_interaction_contract())


def native_item_interaction_v3_contract_sha256() -> str:
    return _canonical_sha256(native_item_interaction_v3_contract())


__all__ = [
    "INTERACTION_TYPE_NAMES",
    "NATIVE_ITEM_CHARGE_TIME_CAPACITY",
    "NATIVE_ITEM_BLOCK_CHANGE_CAPACITY",
    "NATIVE_ITEM_EDGE_CAPACITY",
    "NATIVE_ITEM_INTERACTION_CAPACITY",
    "NATIVE_ITEM_INTERACTION_SCHEMA",
    "NATIVE_ITEM_INTERACTION_SCHEMA_V2",
    "NATIVE_ITEM_INTERACTION_SCHEMA_V3",
    "NATIVE_ITEM_INTERACTION_VERSION",
    "NATIVE_ITEM_INTERACTION_VERSION_V3",
    "NATIVE_ITEM_METADATA_CAPACITY",
    "NATIVE_ITEM_TRIGGER_CAPACITY",
    "NativeBreakBlockPayload",
    "NativeChangeBlockPayload",
    "NativeInteractionEdge",
    "NativeInteractionItemPayload",
    "NativeInteractionModeSettings",
    "NativeInteractionNode",
    "NativeInteractionRoot",
    "NativeInteractionRules",
    "NativeInteractionTrigger",
    "NativeItemInteractionEvidence",
    "NativeModifyInventoryPayload",
    "NativePlaceBlockPayload",
    "native_item_interaction_contract",
    "native_item_interaction_contract_sha256",
    "native_item_interaction_v3_contract",
    "native_item_interaction_v3_contract_sha256",
    "native_item_interaction_evidence_request",
]
