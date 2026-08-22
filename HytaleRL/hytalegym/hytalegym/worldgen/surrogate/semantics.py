"""Local 0.5.7 block collision and default-LOS asset semantics."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import product
from typing import Any

from hytalegym.geometry.contract import (
    FLAG_OPAQUE,
    FLAG_PROTRUDES_CELL,
    FLAG_SOLID,
    MAX_DETAIL_BOXES,
)
from hytalegym.worldgen.surrogate.assets import (
    AssetProvenance,
    CollisionBox,
    HytaleAssetArchive,
    HytaleAssetError,
    decode_prefab_filler,
)
from hytalegym.worldgen.block_affordances import (
    resolve_local_block_affordance,
)

_STATE_SEPARATOR = "_State_Definitions_"
_OPACITY_VALUES = {"Cutout", "Semitransparent", "Solid", "Transparent"}
_MATERIAL_VALUES = {"Empty", "Solid"}
_BLOCK_DEFAULTS: dict[str, Any] = {
    "Material": "Empty",
    "Opacity": "Solid",
    "HitboxType": "Full",
    "IsDoor": False,
}


@dataclass(frozen=True)
class LocalBlockSemantics:
    """Resolved local assets; native differential certification is separate."""

    reference: str
    asset_id: str
    state: str | None
    rotation: int
    material: str
    opacity: str
    hitbox_type: str
    flags: int
    collision_boxes: tuple[CollisionBox, ...]
    is_door: bool
    interaction_id: str | None
    use_state_changes: tuple[tuple[str, str], ...]
    inheritance_chain: tuple[str, ...]
    item_provenance: tuple[AssetProvenance, ...]
    hitbox_provenance: AssetProvenance | None
    support: int = 0
    filler: int = 0
    affordance_valid: bool = False
    affordance_tags: int = 0
    gather_type_index: int = 0
    required_tool_quality: int = 0

    @property
    def solid(self) -> bool:
        return bool(self.flags & FLAG_SOLID)

    @property
    def opaque(self) -> bool:
        return bool(self.flags & FLAG_OPAQUE)


class LocalBlockSemanticsResolver:
    """Resolve item inheritance, state variants, and rotated detail boxes."""

    def __init__(
        self,
        archive: HytaleAssetArchive,
        *,
        inheritance_capacity: int = 32,
    ):
        if not isinstance(archive, HytaleAssetArchive):
            raise TypeError("archive must be a HytaleAssetArchive")
        if (
            isinstance(inheritance_capacity, bool)
            or not isinstance(inheritance_capacity, int)
            or inheritance_capacity <= 0
        ):
            raise ValueError("inheritance capacity must be a positive integer")
        self.archive = archive
        self.inheritance_capacity = inheritance_capacity
        self._cache: dict[
            str,
            tuple[dict[str, Any], tuple[AssetProvenance, ...]],
        ] = {}

    def resolve(
        self,
        reference: str,
        *,
        rotation: int | None = None,
        filler: int | None = None,
    ) -> LocalBlockSemantics:
        """Resolve one prefab reference, rejecting unresolved mechanics."""

        asset_id, state = _parse_reference(reference)
        rotation_index = _rotation_index(rotation)
        filler_offset = decode_prefab_filler(filler)
        filler_value = 0 if filler is None else filler
        if asset_id == "Empty":
            if state is not None:
                raise HytaleAssetError("Empty cannot select a block state")
            return LocalBlockSemantics(
                reference=reference,
                asset_id=asset_id,
                state=None,
                rotation=rotation_index,
                material="Empty",
                opacity="Transparent",
                hitbox_type="Empty",
                flags=0,
                collision_boxes=(),
                is_door=False,
                interaction_id=None,
                use_state_changes=(),
                inheritance_chain=(),
                item_provenance=(),
                hitbox_provenance=None,
                filler=filler_value,
            )

        item, provenance, block = self._resolve_item_block(asset_id, state)
        connected = block.get("ConnectedBlockRuleSet")
        if connected is not None and not _connected_geometry_resolved(
            connected,
            asset_id,
            state,
        ):
            raise HytaleAssetError(
                "connected-block geometry requires an explicit resolved state"
            )

        material = _enum_string(
            block.get("Material"),
            _MATERIAL_VALUES,
            "block material",
        )
        opacity = _enum_string(
            block.get("Opacity"),
            _OPACITY_VALUES,
            "block opacity",
        )
        hitbox_type = _required_string(
            block.get("HitboxType"),
            "block hitbox type",
        )
        if hitbox_type == "Full":
            boxes = (CollisionBox((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),)
            hitbox_provenance = None
        else:
            boxes, hitbox_provenance = self.archive.load_hitbox_type(hitbox_type)
        if len(boxes) > MAX_DETAIL_BOXES:
            raise HytaleAssetError(
                f"hitbox exceeds detail-box capacity {MAX_DETAIL_BOXES}"
            )
        rotated_boxes = tuple(_rotate_box(box, rotation_index) for box in boxes)
        translated_boxes = tuple(
            _translate_box(
                box,
                tuple(-value for value in filler_offset),
            )
            for box in rotated_boxes
        )

        flags = 0
        if material == "Solid":
            flags |= FLAG_SOLID
        if opacity != "Transparent":
            flags |= FLAG_OPAQUE
        if _protrudes_unit_box(translated_boxes):
            flags |= FLAG_PROTRUDES_CELL
        is_door = block.get("IsDoor")
        if not isinstance(is_door, bool):
            raise HytaleAssetError("BlockType.IsDoor must be boolean")
        interaction_id = None
        use_state_changes: tuple[tuple[str, str], ...] = ()
        interactions = block.get("Interactions")
        if interactions is not None:
            if not isinstance(interactions, dict):
                raise HytaleAssetError("BlockType.Interactions must be an object")
            use = interactions.get("Use")
            if isinstance(use, str):
                interaction_id = _required_string(
                    use,
                    "block use interaction",
                )
            elif use is not None and is_door:
                raise HytaleAssetError("door use interaction must be a string")
            elif use is not None:
                use_state_changes = _simple_change_state_use(use)
                if use_state_changes:
                    interaction_id = "ChangeState"
        if is_door and interaction_id is None:
            raise HytaleAssetError("door block has no use interaction")
        try:
            affordance = resolve_local_block_affordance(item, block)
        except (TypeError, ValueError) as error:
            raise HytaleAssetError(
                f"block asset {asset_id!r} has invalid affordances"
            ) from error
        return LocalBlockSemantics(
            reference=reference,
            asset_id=asset_id,
            state=state,
            rotation=rotation_index,
            material=material,
            opacity=opacity,
            hitbox_type=hitbox_type,
            flags=flags,
            collision_boxes=translated_boxes if material == "Solid" else (),
            is_door=is_door,
            interaction_id=interaction_id,
            use_state_changes=use_state_changes,
            inheritance_chain=tuple(item.entry_path for item in provenance),
            item_provenance=provenance,
            hitbox_provenance=hitbox_provenance,
            filler=filler_value,
            affordance_valid=affordance.valid,
            affordance_tags=affordance.tags,
            gather_type_index=affordance.gather_type_index,
            required_tool_quality=affordance.required_tool_quality,
        )

    def resolve_item_asset(
        self,
        reference: str,
    ) -> tuple[dict[str, Any], tuple[AssetProvenance, ...]]:
        """Return inherited item data with an optional block state applied."""

        asset_id, state = _parse_reference(reference)
        if asset_id == "Empty":
            raise HytaleAssetError("Empty has no item asset")
        item, provenance, block = self._resolve_item_block(asset_id, state)
        result = copy.deepcopy(item)
        result["BlockType"] = block
        return result, provenance

    def load_inherited_item_asset(
        self,
        asset_id: str,
    ) -> tuple[dict[str, Any], tuple[AssetProvenance, ...]]:
        """Return inherited data for one ordinary item asset."""

        identifier, state = _parse_reference(asset_id)
        if identifier == "Empty" or state is not None:
            raise HytaleAssetError(
                "ordinary item lookup requires a non-state asset ID"
            )
        item, provenance = self._load_inherited(identifier, ())
        return copy.deepcopy(item), provenance

    def _resolve_item_block(
        self,
        asset_id: str,
        state: str | None,
    ) -> tuple[
        dict[str, Any],
        tuple[AssetProvenance, ...],
        dict[str, Any],
    ]:
        item, provenance = self._load_inherited(asset_id, ())
        block_value = item.get("BlockType")
        if not isinstance(block_value, dict):
            raise HytaleAssetError(
                f"item asset {asset_id!r} does not define BlockType"
            )
        block = _merge_block_type(_BLOCK_DEFAULTS, block_value)
        if state is None:
            return item, provenance, block
        state_root = block.get("State")
        if not isinstance(state_root, dict):
            raise HytaleAssetError(f"block asset {asset_id!r} defines no states")
        definitions = state_root.get("Definitions")
        if not isinstance(definitions, dict) or state not in definitions:
            raise HytaleAssetError(
                f"block state {state!r} is not defined by {asset_id!r}"
            )
        state_value = definitions[state]
        if not isinstance(state_value, dict):
            raise HytaleAssetError(f"block state {state!r} is malformed")
        return item, provenance, _merge_block_type(block, state_value)

    def _load_inherited(
        self,
        asset_id: str,
        stack: tuple[str, ...],
    ) -> tuple[dict[str, Any], tuple[AssetProvenance, ...]]:
        cached = self._cache.get(asset_id)
        if cached is not None:
            return cached
        if asset_id in stack:
            raise HytaleAssetError(
                f"item inheritance cycle: {' -> '.join(stack + (asset_id,))}"
            )
        if len(stack) >= self.inheritance_capacity:
            raise HytaleAssetError("item inheritance exceeds capacity")
        data, provenance = self.archive.load_item_asset(asset_id)
        parent = data.get("Parent")
        child = {key: value for key, value in data.items() if key != "Parent"}
        if parent is None:
            result = copy.deepcopy(child)
            chain = (provenance,)
        else:
            parent_id = _required_string(parent, "item Parent")
            inherited, parent_chain = self._load_inherited(
                parent_id,
                stack + (asset_id,),
            )
            result = _merge_item_asset(inherited, child)
            chain = parent_chain + (provenance,)
        self._cache[asset_id] = (result, chain)
        return result, chain


def _parse_reference(reference: str) -> tuple[str, str | None]:
    value = _required_string(reference, "block asset reference")
    if not value.startswith("*"):
        if "/" in value or "\\" in value:
            raise HytaleAssetError("block asset reference is unsafe")
        return value, None
    encoded = value[1:]
    asset_id, separator, state = encoded.partition(_STATE_SEPARATOR)
    if not separator or not asset_id or not state:
        raise HytaleAssetError("malformed block state asset reference")
    if any(character in asset_id for character in "/\\"):
        raise HytaleAssetError("block state asset reference is unsafe")
    return asset_id, state


def _simple_change_state_use(value: Any) -> tuple[tuple[str, str], ...]:
    """Decode only the exact one-node 0.5.7 ChangeState interaction subset."""

    if not isinstance(value, dict) or set(value) != {"Interactions"}:
        return ()
    nodes = value["Interactions"]
    if not isinstance(nodes, list) or len(nodes) != 1:
        return ()
    node = nodes[0]
    if not isinstance(node, dict) or node.get("Type") != "ChangeState":
        return ()
    if set(node).difference({"Type", "Changes", "UpdateBlockState"}):
        return ()
    update_block_state = node.get("UpdateBlockState", False)
    if not isinstance(update_block_state, bool):
        raise HytaleAssetError("ChangeState.UpdateBlockState must be boolean")
    if update_block_state:
        return ()
    changes = node.get("Changes")
    if not isinstance(changes, dict) or not changes:
        raise HytaleAssetError("ChangeState.Changes must be a non-empty object")
    return tuple(
        (
            _required_string(source, "ChangeState source state"),
            _required_string(target, "ChangeState target state"),
        )
        for source, target in changes.items()
    )


def _connected_geometry_resolved(
    value: Any,
    asset_id: str,
    state: str | None,
) -> bool:
    if state is not None:
        return True
    if not isinstance(value, dict):
        return False
    rule_type = value.get("Type")
    if rule_type == "CustomTemplate":
        patterns = value.get("TemplateShapeBlockPatterns")
        return (
            isinstance(patterns, dict)
            and bool(patterns)
            and all(block == asset_id for block in patterns.values())
        )
    if rule_type == "Stair":
        return _is_literal_default_output(value.get("Straight"), asset_id)
    if rule_type == "Roof":
        regular = value.get("Regular")
        return isinstance(regular, dict) and _is_literal_default_output(
            regular.get("Straight"),
            asset_id,
        )
    return False


def _is_literal_default_output(value: Any, asset_id: str) -> bool:
    """Prove that a connected rule's authored base cell is its straight cell."""

    if not isinstance(value, dict) or set(value).difference({"State", "Block"}):
        return False
    return value.get("State") in (None, "default") and value.get("Block") in (
        None,
        asset_id,
    )


def _rotation_index(value: int | None) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("block rotation must be an integer")
    if value < 0 or value >= 64:
        raise HytaleAssetError("block rotation exceeds RotationTuple capacity")
    return value


def _deep_merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _merge_item_asset(
    parent: dict[str, Any],
    child: dict[str, Any],
) -> dict[str, Any]:
    """Merge raw item data while respecting the contained BlockType codec."""

    result = _deep_merge(parent, child)
    parent_block = parent.get("BlockType")
    child_block = child.get("BlockType")
    if isinstance(parent_block, dict) and isinstance(child_block, dict):
        result["BlockType"] = _merge_block_type(parent_block, child_block)
    return result


def _merge_block_type(
    parent: dict[str, Any],
    child: dict[str, Any],
) -> dict[str, Any]:
    """Apply the non-recursive boundaries inside BlockType.CODEC."""

    result = _deep_merge(parent, child)
    child_gathering = child.get("Gathering")
    if isinstance(child_gathering, dict):
        parent_gathering = parent.get("Gathering")
        inherited_default = (
            parent_gathering.get("UseDefaultDropWhenPlaced")
            if isinstance(parent_gathering, dict)
            else None
        )
        gathering = copy.deepcopy(child_gathering)
        if (
            "UseDefaultDropWhenPlaced" not in gathering
            and inherited_default is not None
        ):
            gathering["UseDefaultDropWhenPlaced"] = copy.deepcopy(
                inherited_default
            )
        result["Gathering"] = gathering
    return result


def _rotate_box(box: CollisionBox, rotation: int) -> CollisionBox:
    yaw = rotation & 3
    pitch = (rotation >> 2) & 3
    roll = (rotation >> 4) & 3
    corners = []
    for x, y, z in product(
        (box.minimum[0], box.maximum[0]),
        (box.minimum[1], box.maximum[1]),
        (box.minimum[2], box.maximum[2]),
    ):
        if roll == 1:
            x, y = 1.0 - y, x
        elif roll == 2:
            x, y = 1.0 - x, 1.0 - y
        elif roll == 3:
            x, y = y, 1.0 - x
        if pitch == 1:
            y, z = 1.0 - z, y
        elif pitch == 2:
            y, z = 1.0 - y, 1.0 - z
        elif pitch == 3:
            y, z = z, 1.0 - y
        if yaw == 1:
            x, z = z, 1.0 - x
        elif yaw == 2:
            x, z = 1.0 - x, 1.0 - z
        elif yaw == 3:
            x, z = 1.0 - z, x
        corners.append((x, y, z))
    return CollisionBox(
        minimum=tuple(min(point[axis] for point in corners) for axis in range(3)),
        maximum=tuple(max(point[axis] for point in corners) for axis in range(3)),
    )


def _protrudes_unit_box(boxes: tuple[CollisionBox, ...]) -> bool:
    return any(
        any(value < 0.0 for value in box.minimum)
        or any(value > 1.0 for value in box.maximum)
        for box in boxes
    )


def _translate_box(
    box: CollisionBox,
    offset: tuple[int, int, int],
) -> CollisionBox:
    return CollisionBox(
        minimum=tuple(box.minimum[axis] + offset[axis] for axis in range(3)),
        maximum=tuple(box.maximum[axis] + offset[axis] for axis in range(3)),
    )


def _enum_string(value: Any, allowed: set[str], label: str) -> str:
    result = _required_string(value, label)
    if result not in allowed:
        raise HytaleAssetError(f"unsupported {label} {result!r}")
    return result


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise HytaleAssetError(f"{label} must be a non-empty string")
    return value


__all__ = [
    "LocalBlockSemantics",
    "LocalBlockSemanticsResolver",
]
